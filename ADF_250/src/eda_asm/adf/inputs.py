"""ADF EDA-NOCV input deck generator.

Given a reaction with R/TS/P geometries plus explicit fragmentation
(atom indices for fragment A and B into the TS atom array), generate a
self-contained bash .run script that does:

  1. SP of fragment A at TS-frozen geometry          → fragA_at_TS.results/
  2. SP of fragment B at TS-frozen geometry          → fragB_at_TS.results/
  3. EDA-NOCV at full TS geometry using above frags  → eda_TS.results/
  4. SP of fragment A at relaxed (R) geometry        → fragA_relaxed.results/
  5. SP of fragment B at relaxed (P) geometry        → fragB_relaxed.results/

The 5-vector ASR label is then:
  ΔE_Pauli, ΔV_elst, ΔE_orb, ΔE_disp   ← from EDA step (3) on eda_TS output
  ΔE_strain = (E[fragA_at_TS] - E[fragA_relaxed]) + (E[fragB_at_TS] - E[fragB_relaxed])

Settings follow CLAUDE.md:
  Functional PBE0 + D3(BJ)
  Basis TZ2P, Core=None
  NumericalQuality Good
  Symmetry NOSYM
  Scalar ZORA only when Br is present.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ase.data
import numpy as np

AMSBASHRC_DEFAULT = "/home1/yeseo1ee/ams2026.103/amsbashrc.sh"


@dataclass
class FragmentSpec:
    name: str                              # "fA" or "fB"
    indices: np.ndarray                    # indices into the full TS atom array (int)
    numbers: np.ndarray                    # atomic numbers (int)
    positions_at_TS: np.ndarray            # (n, 3) coordinates sliced from TS geometry
    positions_relaxed: np.ndarray          # (n, 3) coordinates from relaxed reference
    charge: int = 0


@dataclass
class ADFRunSpec:
    reaction_id: str
    fragA: FragmentSpec
    fragB: FragmentSpec
    ts_numbers: np.ndarray                 # full system atomic numbers
    ts_positions: np.ndarray               # full system at TS (Å)
    total_charge: int = 0
    amsbashrc: str = AMSBASHRC_DEFAULT
    functional: str = "PBE0"               # hybrid GGA
    dispersion: str = "Grimme3 BJDAMP"     # D3(BJ)
    basis_type: str = "TZ2P"
    basis_core: str = "None"
    numerical_quality: str = "Good"
    extra_provenance: dict = field(default_factory=dict)

    @property
    def has_Br(self) -> bool:
        return 35 in set(int(z) for z in self.ts_numbers.tolist())


def _fmt_atom(symbol: str, x: float, y: float, z: float, frag_label: str | None = None) -> str:
    suf = f"  adf.f={frag_label}" if frag_label else ""
    return f"    {symbol:<3s} {x: .8f} {y: .8f} {z: .8f}{suf}"


def _engine_block(spec: ADFRunSpec, *, fragments_ref: dict | None = None,
                  etsnocv: bool = False) -> str:
    lines = [
        "  Basis",
        f"    Type {spec.basis_type}",
        f"    Core {spec.basis_core}",
        "  End",
        "  XC",
        f"    Hybrid {spec.functional}",
        f"    Dispersion {spec.dispersion}",
        "  End",
        f"  NumericalQuality {spec.numerical_quality}",
        "  Symmetry NOSYM",
    ]
    if spec.has_Br:
        lines += [
            "  Relativity",
            "    Level Scalar",
            "    Formalism ZORA",
            "  End",
        ]
    if fragments_ref:
        lines.append("  Fragments")
        for label, ref in fragments_ref.items():
            lines.append(f"    {label} {ref}")
        lines.append("  End")
    if etsnocv:
        lines += [
            "  ETSNOCV",
            "    EKMIN 0.5",
            "    ENOCV 0.05",
            "  End",
            "  Print EtsLowdin",
        ]
    return "\n".join(lines)


def _sp_block(jobname: str, numbers: np.ndarray, positions: np.ndarray, *,
              charge: int, spec: ADFRunSpec) -> str:
    atom_lines = [
        _fmt_atom(ase.data.chemical_symbols[int(z)], float(x), float(y), float(zc))
        for z, (x, y, zc) in zip(numbers, positions)
    ]
    engine = _engine_block(spec)
    return (
        f'AMS_JOBNAME={jobname} "$AMSBIN/ams" > {jobname}.out 2>&1 << \'EOF\'\n'
        "System\n"
        "  atoms\n"
        + "\n".join(atom_lines) + "\n"
        "  end\n"
        f"  charge {charge}\n"
        "end\n"
        "Task SinglePoint\n"
        "Engine ADF\n"
        f"  title {jobname} for {spec.reaction_id}\n"
        f"{engine}\n"
        "EndEngine\n"
        "EOF\n"
    )


def _eda_block(spec: ADFRunSpec) -> str:
    fA_idx = set(int(i) for i in spec.fragA.indices.tolist())
    fB_idx = set(int(i) for i in spec.fragB.indices.tolist())
    atom_lines = []
    for i, (z, (x, y, zc)) in enumerate(zip(spec.ts_numbers, spec.ts_positions)):
        sym = ase.data.chemical_symbols[int(z)]
        if i in fA_idx:
            label = "fA"
        elif i in fB_idx:
            label = "fB"
        else:
            raise ValueError(
                f"TS atom {i} not assigned to a fragment (reaction {spec.reaction_id})"
            )
        atom_lines.append(_fmt_atom(sym, float(x), float(y), float(zc),
                                    frag_label=label))
    engine = _engine_block(
        spec,
        fragments_ref={
            "fA": "fragA_at_TS.results/adf.rkf",
            "fB": "fragB_at_TS.results/adf.rkf",
        },
        etsnocv=True,
    )
    return (
        'AMS_JOBNAME=eda_TS "$AMSBIN/ams" > eda_TS.out 2>&1 << \'EOF\'\n'
        "System\n"
        "  atoms\n"
        + "\n".join(atom_lines) + "\n"
        "  end\n"
        f"  charge {spec.total_charge}\n"
        "end\n"
        "Task SinglePoint\n"
        "Engine ADF\n"
        f"  title EDA-NOCV at TS for {spec.reaction_id}\n"
        f"{engine}\n"
        "EndEngine\n"
        "EOF\n"
    )


def _write_xyz(path: Path, numbers: np.ndarray, positions: np.ndarray,
               comment: str = "") -> None:
    n = int(len(numbers))
    lines = [str(n), comment]
    for z, (x, y, zc) in zip(numbers, positions):
        sym = ase.data.chemical_symbols[int(z)]
        lines.append(f"{sym:<3s} {x: .8f} {y: .8f} {zc: .8f}")
    path.write_text("\n".join(lines) + "\n")


def generate_run_script(spec: ADFRunSpec, out_dir: Path) -> Path:
    """Write run_eda.sh, geometry XYZ snapshots, and meta.json to out_dir.

    Returns the path to the run script (chmod +x).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_xyz(out_dir / "TS.xyz", spec.ts_numbers, spec.ts_positions,
               comment=f"TS for {spec.reaction_id}")
    _write_xyz(out_dir / "fragA_at_TS.xyz", spec.fragA.numbers,
               spec.fragA.positions_at_TS,
               comment=f"fragA at TS for {spec.reaction_id}")
    _write_xyz(out_dir / "fragB_at_TS.xyz", spec.fragB.numbers,
               spec.fragB.positions_at_TS,
               comment=f"fragB at TS for {spec.reaction_id}")
    _write_xyz(out_dir / "fragA_relaxed.xyz", spec.fragA.numbers,
               spec.fragA.positions_relaxed,
               comment=f"fragA relaxed for {spec.reaction_id}")
    _write_xyz(out_dir / "fragB_relaxed.xyz", spec.fragB.numbers,
               spec.fragB.positions_relaxed,
               comment=f"fragB relaxed for {spec.reaction_id}")

    blocks = [
        _sp_block("fragA_at_TS", spec.fragA.numbers, spec.fragA.positions_at_TS,
                  charge=spec.fragA.charge, spec=spec),
        _sp_block("fragB_at_TS", spec.fragB.numbers, spec.fragB.positions_at_TS,
                  charge=spec.fragB.charge, spec=spec),
        _eda_block(spec),
        _sp_block("fragA_relaxed", spec.fragA.numbers, spec.fragA.positions_relaxed,
                  charge=spec.fragA.charge, spec=spec),
        _sp_block("fragB_relaxed", spec.fragB.numbers, spec.fragB.positions_relaxed,
                  charge=spec.fragB.charge, spec=spec),
    ]
    body = "\n".join(blocks)

    script = f"""#!/bin/bash
# ADF EDA-NOCV pipeline for reaction {spec.reaction_id}
# Functional: Hybrid {spec.functional} + Dispersion {spec.dispersion}
# Basis     : {spec.basis_type} (Core={spec.basis_core})
# NumQuality: {spec.numerical_quality}
# Symmetry  : NOSYM
# Relativity: {'Scalar ZORA' if spec.has_Br else 'none (no Br)'}
# Total charge: {spec.total_charge}
set -eo pipefail
# amsbashrc.sh probes $SCMLICENSE via `test -z` which fails under `set -u`,
# so source it before enabling strict-undefined mode.
source {spec.amsbashrc}
set -u
cd "$(dirname "$0")"
# AMS picks up NSCM as the number of MPI ranks; we map to SLURM_NTASKS
# (since AMS uses `srun -n $NSCM` internally on SLURM systems).
export NSCM=${{SLURM_NTASKS:-${{SLURM_CPUS_PER_TASK:-1}}}}
ulimit -s unlimited 2>/dev/null || true
echo "=== reaction={spec.reaction_id} host=$(hostname) date=$(date -Iseconds) NSCM=$NSCM ==="

{body}

echo "=== reaction={spec.reaction_id} all 5 SPs completed at $(date -Iseconds) ==="
"""
    out_path = out_dir / "run_eda.sh"
    out_path.write_text(script)
    out_path.chmod(0o755)
    return out_path
