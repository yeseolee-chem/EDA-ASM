"""spec19 Stage 2 aggregator — summary.md, DEVIATIONS.md (append #7 + #8), figures."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec19_espley_s2_structures"
MANIFEST = STAGE / "results/manifest.pkl"
GATES_LOG = STAGE / "logs/gates.log"

OUT_SUM = STAGE / "results/summary.md"
OUT_DEV = STAGE / "results/DEVIATIONS.md"
FIG_NATOMS = STAGE / "figures/natoms_hist.png"
FIG_FSIZE = STAGE / "figures/fragment_size_scatter.png"

DEVIATIONS = [
    ("1", "ORCA replaces Gaussian 16", "3",
     "user-mandated; no Gaussian in project stack for this line"),
    ("2", "GFN2-xTB replaces AM1 as the SQM level", "3",
     "ORCA has no AM1; GFN2 is the nearest SQM-tier substitute"),
    ("3", "q_barrier (ΔG‡) omitted from both targets and features", "1, 4",
     "electronic-energy consistency (SPEC_14); 44 features vs their 45"),
    ("4", "Reference DFT is ORCA ωB97X-3c EDA-NOCV, not B3LYP-D3(BJ)/def2-TZVP + SMD", "1",
     "our labels; must be noted on every cross-paper kcal/mol comparison"),
    ("5", "Cohort is 400 dipolar [3+2] cycloadditions, not 3510", "1",
     "user-requested restriction to the 400-set (192 locked_778 + 208 spec16)"),
    ("6", "f_select.py line ~226 substring `am1` → `am1|gfn2` (one-line patch required)", "3",
     "Rev 1 G-E confirmed `_gfn2` targets are dropped by the unmodified _manual_runner"),
    ("7", "diassep.py NOT used — TS fragment partition inherited from EDA-NOCV label pipeline", "2",
     "User-mandated: fragment A/B split follows the (1)/(2) labels in eda.inp verbatim. "
     "Re-deriving from imaginary-mode displacements would risk a split inconsistent with "
     "the strain_A/strain_B labels fixed at Stage 1"),
    ("8", "locked_778 r_A/r_B are R.xyz atom subsets, NOT independently-optimized isolated fragments", "2",
     "The 192 locked_778 reactions inherited relaxed-fragment geometries from the "
     "reactant-complex (R.xyz), which is what strain_A_kcal was computed against. "
     "The 208 spec16 reactions use fully-optimized opt.xyz. Both are consistent with "
     "their own strain-label semantics; the two halves have different DIAS definitions "
     "of 'relaxed reactant'."),
]


def write_deviations() -> None:
    lines = [
        "# DEVIATIONS — Espley et al. replication line", "",
        "Started at Stage 1 (spec18r1_espley_s1_labels_fix). Every downstream "
        "stage inherits and appends to this list.", "",
        "| # | deviation | stage | rationale |",
        "|---|---|---|---|",
    ]
    for r in DEVIATIONS:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
    lines += [
        "",
        "## Fragment A/B convention (INVARIANT — INHERITED, NEVER RE-DERIVED)",
        "",
        "Fragment A = ORCA EDA `(1)` atoms in `eda.inp`  =  `strain_A_kcal`  =  "
        "Stage-1 dict key `{reaction_number}_1`.",
        "",
        "Fragment B = ORCA EDA `(2)` atoms in `eda.inp`  =  `strain_B_kcal`  =  "
        "Stage-1 dict key `{reaction_number}_2`.",
        "",
        "This is the user-mandated convention. No stage of this pipeline is "
        "permitted to derive fragmentation from independent methods (Svatunek "
        "displacement analysis, adjacency-graph reasoning, RDKit fragment "
        "detection, etc.). Downstream anomalies MUST be reported, never "
        "silently corrected.",
    ]
    OUT_DEV.write_text("\n".join(lines) + "\n")


def natoms_hist(mf: pd.DataFrame) -> None:
    STAGE.joinpath("figures").mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key in zip(axes, ["ts", "r_A", "r_B"]):
        vals = np.array([r[key] for r in mf["natoms"]])
        ax.hist(vals, bins=range(int(vals.min()), int(vals.max()) + 2),
                color="#3b7dbf", edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.set_xlabel(f"n_atoms({key})")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIG_NATOMS, dpi=140)


def fragment_size_scatter(mf: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    xA = np.array([r["r_A"] for r in mf["natoms"]])
    xB = np.array([r["r_B"] for r in mf["natoms"]])
    color_map = {"locked_778": "#3b7dbf", "spec16": "#e07b00"}
    for sub in mf["sub_source"].unique():
        mask = mf["sub_source"] == sub
        ax.scatter(xA[mask], xB[mask], s=18, alpha=0.65, color=color_map.get(sub, "gray"),
                   label=f"{sub} (n={int(mask.sum())})", edgecolor="none")
    ax.set_xlabel("n_atoms(fragment A)")
    ax.set_ylabel("n_atoms(fragment B)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_FSIZE, dpi=140)


def summarise_gates() -> tuple[int, int, str]:
    text = GATES_LOG.read_text() if GATES_LOG.exists() else ""
    n_fail = text.count("FAIL ")
    n_warn = text.count("WARN ")
    lines = [ln for ln in text.splitlines() if any(t in ln for t in
             ("PASS", "FAIL", "WARN", "INFO", "SUMMARY"))]
    return n_fail, n_warn, "\n".join(f"    {ln}" for ln in lines)


def manifest_sha256() -> str:
    h = hashlib.sha256()
    with open(MANIFEST, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_summary(mf: pd.DataFrame) -> None:
    n_fail, n_warn, digest = summarise_gates()
    manifest_sha = manifest_sha256()
    anomalies_p = STAGE / "results/common_atom_anomalies.csv"
    openshell_p = STAGE / "results/open_shell.csv"
    diassep_p = STAGE / "results/diassep_agreement.csv"

    def _sz(p):
        try:
            return len(pd.read_csv(p)) if p.exists() else 0
        except Exception:
            return 0
    n_anomalies = _sz(anomalies_p)
    n_openshell = _sz(openshell_p)
    n_diassep = _sz(diassep_p)

    lines = []
    lines.append("# spec19_espley_s2_structures — Stage 2 summary (DIPOLAR-400)")
    lines.append("")
    lines.append("Assembles the 5 DIAS structures per reaction (r_A, r_B, ts, "
                 "d_A, d_B) on the 400 dipolar [3+2] cycloaddition cohort, plus "
                 "the three bookkeeping artifacts (manifest, mapping, mol_types).")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- python: {platform.python_version()}")
    lines.append(f"- pandas: {pd.__version__}")
    lines.append("")
    lines.append("**G2-0 (pandas round-trip):** Stage 1 pickle loaded cleanly in "
                 "reactot (pandas 2.3.3). The `espley_repro` env (pandas 2.1.1) is "
                 "not present on this HPC; a round-trip smoke test in that env is "
                 "DEFERRED and must run before Stage 5 consumes the artifact.")
    lines.append("")

    lines.append("## Fragment split (USER-MANDATED, INHERITED)")
    lines.append("")
    lines.append("Fragment A = ORCA EDA `(1)` atoms → `strain_A_kcal` → "
                 "Stage-1 dict key `{rn}_1`.")
    lines.append("Fragment B = ORCA EDA `(2)` atoms → `strain_B_kcal` → "
                 "Stage-1 dict key `{rn}_2`.")
    lines.append("")
    lines.append("Never re-derived. `diassep.py` is not used. Any disagreement with "
                 "external partition methods is a finding to surface, not a defect "
                 "to fix. See Deviation #7.")
    lines.append("")

    lines.append("## Cohort composition")
    lines.append("")
    lines.append("| sub_source | n | r_A source |")
    lines.append("|---|---:|---|")
    lines.append("| locked_778 | 192 | R.xyz atom subset (Deviation #8) |")
    lines.append("| spec16 | 208 | opt.xyz (isolated frag opt at BLYP-D3BJ/def2-TZVP) |")
    lines.append(f"| **total** | **{len(mf)}** | — |")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `results/manifest.pkl` (sha256 `{manifest_sha[:16]}…`)")
    lines.append("  Per-row: `dir`, `natoms`, `charge`, `mult`, `ts_idx_A`, `ts_idx_B`, "
                 "`r_A_provenance`, `r_B_provenance`.")
    lines.append("- `results/common_atoms.pkl` — {rn → {r_A_k, r_B_k, ts_k, d_A_k, d_B_k, "
                 "reacting_A/B_map_ids}}. Reacting-atom counts per the Espley "
                 "(3,2,5,3,2) contract; index enumeration deferred to Stage 4 "
                 "where SMILES↔xyz atom order matching happens.")
    lines.append("- `results/mapping.pkl` — {rn → {ts_idx_A, ts_idx_B, n_atoms}}.")
    lines.append("- `results/mol_types.pkl` — {rn → {A: 'dipole'|'dipolarophile', B: …}}.")
    lines.append(f"- `structures/rxn_XXXX/{{r_A,r_B,ts,d_A,d_B}}.xyz` — 2000 files, "
                 f"NOT committed (large + easily regenerated). "
                 "On-HPC location: "
                 f"`{STAGE}/structures/`.")
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append(f"**{n_fail} FAIL, {n_warn} WARN** — see `logs/gates.log`.")
    lines.append("")
    lines.append("- G2-0: Stage 1 pickle round-trip in reactot; espley_repro test deferred.")
    lines.append("- G2-A: 2000 xyz files present, no zero-byte, no NaN coords.")
    lines.append("- G2-B: `natoms(ts) == natoms(d_A) + natoms(d_B)`; element multisets "
                 "of `d_A`↔`r_A` and `d_B`↔`r_B` identical.")
    lines.append(f"- G2-C: `d_A ∪ d_B` coordinates are exact atom subsets of `ts` (tolerance 1e-6 Å).")
    lines.append("- G2-D: Fragment-A ↔ dict-key `_1` contract holds by construction; "
                 "manifest records `r_A/r_B` provenance for every row.")
    lines.append(f"- G2-E: reacting-atom k-shape check vs Espley (3,2,5,3,2). "
                 f"{n_anomalies} anomalies logged (report-only, no exclusion).")
    lines.append(f"- G2-F: charge conservation + open-shell list "
                 f"({n_openshell} rxns with fragment mult ≠ 1).")
    lines.append(f"- G2-G: diassep cross-check informational; "
                 f"{n_diassep} rows in `results/diassep_agreement.csv`.")
    lines.append("")
    lines.append("### Gate digest")
    lines.append("```")
    lines.append(digest)
    lines.append("```")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("```")
    lines.append("Ref Comparison/spec19_espley_s2_structures/")
    lines.append("  code/{discover_geometry_sources.py, build_structures.py,")
    lines.append("        build_common_atoms.py, verify_structures.py,")
    lines.append("        diassep_crosscheck.py, aggregate.py, submit_s2.sh}")
    lines.append("  data/                                (empty — cohort_notes inherited)")
    lines.append("  logs/{discovery.json, build.log, gates.log}")
    lines.append("  results/{manifest.pkl, common_atoms.pkl, mapping.pkl, mol_types.pkl,")
    lines.append("           common_atom_anomalies.csv, open_shell.csv,")
    lines.append("           diassep_agreement.csv, DEVIATIONS.md, summary.md}")
    lines.append("  figures/{natoms_hist.png, fragment_size_scatter.png}")
    lines.append("  structures/rxn_XXXX/{r_A,r_B,ts,d_A,d_B}.xyz   (NOT committed)")
    lines.append("```")
    lines.append("")

    OUT_SUM.write_text("\n".join(lines) + "\n")


def main() -> None:
    mf = pd.read_pickle(MANIFEST)
    natoms_hist(mf)
    fragment_size_scatter(mf)
    write_deviations()
    write_summary(mf)


if __name__ == "__main__":
    main()
