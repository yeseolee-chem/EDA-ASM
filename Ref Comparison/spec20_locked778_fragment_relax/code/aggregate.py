"""spec20 aggregator — writes summary.md + DEVIATIONS.md from
protocol_discovery.json.

Emits an at-a-glance table + a plain statement of the G20-0 outcome
(pass, halt with cross-half divergence, halt with within-half
inconsistency, or all three). Includes the pre-registered §7 default
action for each halt path.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec20_locked778_fragment_relax"
DISCOVERY = STAGE / "logs/protocol_discovery.json"
HALT_FLAG = STAGE / "logs/G20_0_HALT.flag"
PASS_FLAG = STAGE / "logs/G20_0_PASS.flag"
OUT_SUM = STAGE / "results/summary.md"
OUT_DEV = STAGE / "results/DEVIATIONS.md"


def _profile_row(prof: dict, side: str) -> dict:
    p = prof[side]
    return {
        "n": p["n"],
        "route": next(iter(p["routes"])) if p["routes"] else "",
        "cp": ", ".join(f"{k}={v}" for k, v in p["cp"].items()),
        "opt": ", ".join(f"{k}={v}" for k, v in p["opt"].items()),
        "solvent": ", ".join(f"{k}={v}" for k, v in p["solvent"].items()),
    }


def main() -> int:
    STAGE.joinpath("results").mkdir(parents=True, exist_ok=True)

    with open(DISCOVERY) as jf:
        disc = json.load(jf)
    prof = disc["summary"]["profile_by_half"]
    finding = disc["summary"]["finding"]

    halted = HALT_FLAG.exists()
    passed = PASS_FLAG.exists()

    lines = []
    lines.append("# spec20_locked778_fragment_relax — Stage 3 G20-0 summary")
    lines.append("")
    lines.append("Blocking protocol discovery for the proposed dipolar-192 relaxation. "
                 "Audits the exact ORCA input files that produced `strain_A_kcal` / "
                 "`strain_B_kcal` on all 400 rows.")
    lines.append("")
    lines.append(f"Env: python {platform.python_version()}, pandas {disc['summary']['pandas']}.")
    lines.append("")

    lines.append("## Per-half protocol profile")
    lines.append("")
    for side, label in [("TS_A", "E(TS_A) — fragment A at TS geometry"),
                        ("TS_B", "E(TS_B) — fragment B at TS geometry"),
                        ("R_A",  "E(R_A) — fragment A 'relaxed' reference"),
                        ("R_B",  "E(R_B) — fragment B 'relaxed' reference")]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| half | n | route (first seen) | CP | Opt | solvent |")
        lines.append("|---|---:|---|---|---|---|")
        for half in ("locked_778", "spec16"):
            r = _profile_row(prof[half], side)
            lines.append(f"| {half} | {r['n']} | `{r['route']}` | {r['cp']} | {r['opt']} | {r['solvent']} |")
        lines.append("")

    lines.append("## G20-0 outcome")
    lines.append("")
    if halted:
        lines.append("**G20-0 HALT** — see below.")
        lines.append("")
        lines.append("Per §7 open item 1 the pre-registered default action on any "
                     "cross-half or within-half divergence beyond relaxed-fragment "
                     "geometry is **halt and report**, rather than fix one axis "
                     "while leaving another. The following divergences were found:")
        lines.append("")
        for k, tag in [("cross_half_divergences_beyond_geometry", "Cross-half"),
                       ("within_locked_778_inconsistencies", "Within locked_778"),
                       ("within_spec16_inconsistencies", "Within spec16")]:
            if finding.get(k):
                lines.append(f"### {tag}")
                for d in finding[k]:
                    lines.append(f"- {d}")
                lines.append("")
    elif passed:
        lines.append("**G20-0 PASS** — halves match on all protocol attributes beyond "
                     "relaxed-fragment geometry. Proceed to G20-1 pilot.")
        lines.append("")
    else:
        lines.append("**G20-0 INDETERMINATE** — neither flag was set. See "
                     "`logs/protocol_discovery.json` for raw findings.")
        lines.append("")

    lines.append("## Interpretation and next-step decision")
    lines.append("")
    if halted:
        lines.append("The finding is more nuanced than the spec §7 anticipated:")
        lines.append("")
        lines.append("- **Both halves apply CP correction to E(TS_A) / E(TS_B).** "
                     "The ORCA EDA-NOCV recipe produces per-fragment single points "
                     "at the TS geometry using the paired fragment's basis as "
                     "ghost atoms (visible as `:(1)` / `:(2)` tags in `eda_frag{1,2}.inp`).")
        lines.append("- **The two halves treat E(R_A) / E(R_B) differently.** "
                     "locked_778 uses `v9_review/strain_sp_cp/{rid}/frag{A,B}_R.inp` "
                     "— a CP-corrected single point at the reactant-complex-subset "
                     "geometry. spec16 uses `spec16_orca_strain/inputs/{rid}__f{A,B}/opt.inp` "
                     "— an isolated fragment optimization with NO ghost basis.")
        lines.append("- **This means spec16 is internally inconsistent** on the CP axis: "
                     "TS-side energies are CP-corrected, R-side energies are not. "
                     "locked_778 is internally consistent (CP on both sides).")
        lines.append("")
        lines.append("### What spec20 proposed vs. what the data requires")
        lines.append("")
        lines.append("Spec20 §5 proposes moving locked_778's R-side geometry to match "
                     "spec16's (fully-optimized isolated fragment), leaving the CP "
                     "treatment unchanged. Under the actual data that would:")
        lines.append("")
        lines.append("- Fix the *geometry* axis (both halves would now use isolated-fragment "
                     "R-side geometry).")
        lines.append("- **Leave the CP axis divergent**: locked_778 R-side CP-corrected, "
                     "spec16 R-side not.")
        lines.append("- **Leave the spec16 internal CP asymmetry unchanged**: TS-side "
                     "CP-corrected, R-side not.")
        lines.append("")
        lines.append("Per §7 open item 1 default, **spec20 is halted at G20-0 pending a user decision** among:")
        lines.append("")
        lines.append("1. **Full unification (largest scope)** — recompute strain for all 400 rows under "
                     "a single protocol (either all CP or none). This subsumes spec20 and "
                     "goes beyond it.")
        lines.append("2. **Geometry-only unification** — proceed with spec20 as written, "
                     "accepting the residual CP-axis divergence. Requires documenting a "
                     "new deviation (spec20 fixes #8 partially; a new deviation records "
                     "the residual CP mismatch).")
        lines.append("3. **Fallback to Option B (§7 open item 4)** — restrict the "
                     "downstream training cohort to the 208 spec16 rows, accept spec16's "
                     "own internal CP asymmetry as a documented caveat, and skip the "
                     "relaxation altogether.")
        lines.append("4. **Standby** — halt the Espley replication until the label pipeline "
                     "is redone under a single protocol.")
        lines.append("")
        lines.append("No production compute (the 384-job array) has been submitted.")
        lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("```")
    lines.append("Ref Comparison/spec20_locked778_fragment_relax/")
    lines.append("  code/{discover_protocol.py, aggregate.py, submit_s20.sh}")
    lines.append("  logs/{protocol_discovery.json, discover.log, G20_0_HALT.flag}")
    lines.append("  results/{DEVIATIONS.md, summary.md}")
    lines.append("```")
    lines.append("")

    OUT_SUM.write_text("\n".join(lines) + "\n")
    print(f"[write] {OUT_SUM}")

    # DEVIATIONS delta — spec20 findings, not overwriting Stage 1/2 file
    dev_lines = [
        "# DEVIATIONS — spec20 delta",
        "",
        "This file records what spec20's G20-0 protocol discovery found. It "
        "supplements (does not replace) the cross-stage `DEVIATIONS.md` in "
        "spec19_espley_s2_structures.",
        "",
        "## Update to Deviation #8 (from Stage 2)",
        "",
        "Deviation #8 said the 192 `locked_778` rows use R.xyz atom subsets "
        "(not independently-optimized isolated fragments) as their relaxed-"
        "fragment reference, while the 208 `spec16` rows use fully-optimized "
        "isolated fragments (`opt.xyz`).",
        "",
        "G20-0 confirms this geometric divergence AND reveals a further "
        "protocol asymmetry:",
        "",
        "- Both halves compute E(TS_A), E(TS_B) with counterpoise correction "
        "(`:(1)`/`:(2)` ghost atoms in `eda_frag{1,2}.inp`).",
        "- locked_778 computes E(R_A), E(R_B) with CP correction "
        "(`v9_review/strain_sp_cp/{rid}/frag{A,B}_R.inp` — 7-11 ghost atoms).",
        "- spec16 computes E(R_A), E(R_B) as isolated fragment optimizations "
        "with NO CP (`spec16_orca_strain/inputs/{rid}__f{A,B}/opt.inp`).",
        "",
        "Consequently: **spec16 has an internal CP asymmetry** on strain "
        "(TS-side CP, R-side no CP); **locked_778 is internally consistent** "
        "(CP on both sides).",
        "",
        "## Provisional Deviation #9 (needs user decision at G20-0)",
        "",
        "Whichever unification path the user chooses at G20-0, the CP treatment "
        "of the R-side energies must be equalised before Deviation #8 can be "
        "marked resolved. Options are enumerated in `summary.md` §Interpretation.",
        "",
    ]
    OUT_DEV.write_text("\n".join(dev_lines))
    print(f"[write] {OUT_DEV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
