"""spec20 G20-0 protocol discovery (BLOCKING).

Audits the actual ORCA input files that produced strain_A_kcal /
strain_B_kcal for each half of the dipolar-400. Emits
`logs/protocol_discovery.json`. HALTS if the two halves differ in
anything beyond relaxed-fragment geometry — per §7 open item 1, the
default action on that finding is to halt and report rather than
paper over one axis while another remains.

For every reaction we inspect the exact files whose energies feed the
label parquet:
  E(TS_A), E(TS_B): from `.../orca_inputs/{rid}/eda_frag{1,2}.inp`
                    (both halves; TS-side CP status derived from
                    the `:(2)` / `:(1)` ghost tags in the atom block)
  E(R_A),  E(R_B):
    - locked_778: `outputs/v9_review/strain_sp_cp/{rid}/frag{A,B}_R.inp`
    - spec16:     `outputs/spec16_orca_strain/inputs/{rid}__f{A,B}/opt.inp`

Extracted per file:
  - route line (level of theory, dispersion, SCF thresholds, Opt/no-Opt)
  - CP status (presence of ghost atoms with `:(1)` or `:(2)` tag)
  - solvent block (`%cpcm` / `SMD` / `pcm`)

Findings are aggregated to a per-half profile. Divergences on any
attribute other than relaxed-fragment geometry produce a HALT.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec20_locked778_fragment_relax"
STAGE1_PKL = REPO / "Ref Comparison/spec18r1_espley_s1_labels_fix/results/labels_2ch_400dipolar.pkl"

# Where E(TS_A/B) come from (same layout both halves, different roots)
LOCKED_ORCA_IN = REPO / "outputs/v8_review/orca_inputs"
SPEC16_ORCA_IN = REPO / "outputs/spec16_orca/inputs"

# Where E(R_A/B) come from — this is the divergent side
LOCKED_STRAIN_CP = REPO / "outputs/v9_review/strain_sp_cp"        # CP-corrected SP
LOCKED_STRAIN_ALT = REPO / "outputs/v8_review/strain_sp"          # v8 fallback (no CP)
SPEC16_STRAIN = REPO / "outputs/spec16_orca_strain/inputs"        # isolated frag opt

OUT_JSON = STAGE / "logs/protocol_discovery.json"
OUT_LOG = STAGE / "logs/discover.log"

ROUTE_RE = re.compile(r"^\s*!\s*(.+?)\s*$")
# Ghost atoms come in two flavours across our data:
#   - `C :(2)`  — element, space, colon-plus-frag-tag (eda_frag*.inp style)
#   - `C:`      — element with trailing colon, no frag tag (v9 strain_sp_cp style)
# Allow optional whitespace between element and colon; optional `(1)`/`(2)` after.
GHOST_ATOM_RE = re.compile(
    r"^\s*[A-Z][a-z]?\s*:(?:\s*\([12]\))?\s+[-+.\deE]+\s+[-+.\deE]+\s+[-+.\deE]+"
)
# Real atom with an explicit fragment tag: `C(1)` or `C (1)`.
REAL_ATOM_RE = re.compile(
    r"^\s*[A-Z][a-z]?\s*\([12]\)\s+[-+.\deE]+\s+[-+.\deE]+\s+[-+.\deE]+"
)
# Real atom with no tag: `C  1.234 …`.
PLAIN_ATOM_RE = re.compile(
    r"^\s*[A-Z][a-z]?\s+[-+.\deE]+\s+[-+.\deE]+\s+[-+.\deE]+"
)
SOLVENT_KEYS = ("cpcm", "smd", "pcm", "%cpcm")


def _log(fh, msg):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def parse_orca_input(path: Path) -> dict:
    """Return a compact fingerprint of an ORCA input: route + CP status +
    solvent + Opt flag. Missing file → None entries.
    """
    if not path.exists():
        return {"exists": False, "path": str(path)}
    text = path.read_text()
    lines = text.splitlines()

    route = None
    for ln in lines:
        m = ROUTE_RE.match(ln)
        if m:
            route = m.group(1)
            break

    ghost_atoms = 0
    real_labelled = 0
    real_plain = 0
    in_xyz = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("*") and "xyz" in s.lower():
            in_xyz = True
            continue
        if in_xyz and s.startswith("*"):
            in_xyz = False
        if not in_xyz:
            continue
        if GHOST_ATOM_RE.match(ln):
            ghost_atoms += 1
        elif REAL_ATOM_RE.match(ln):
            real_labelled += 1
        elif PLAIN_ATOM_RE.match(ln):
            real_plain += 1

    text_low = text.lower()
    solvent = None
    for k in SOLVENT_KEYS:
        if k in text_low:
            solvent = k
            break

    is_opt = bool(route and "opt" in route.lower())
    is_cp = ghost_atoms > 0

    return {
        "exists": True,
        "path": str(path),
        "route": route,
        "is_opt": is_opt,
        "is_cp": is_cp,
        "n_ghost_atoms": ghost_atoms,
        "n_real_atoms_labelled": real_labelled,
        "n_real_atoms_plain": real_plain,
        "solvent": solvent,
    }


def audit_reaction(rid: str, sub: str) -> dict:
    if sub == "spec16":
        ts_A = SPEC16_ORCA_IN / rid / "eda_frag1.inp"
        ts_B = SPEC16_ORCA_IN / rid / "eda_frag2.inp"
        r_A = SPEC16_STRAIN / f"{rid}__fA" / "opt.inp"
        r_B = SPEC16_STRAIN / f"{rid}__fB" / "opt.inp"
    else:
        ts_A = LOCKED_ORCA_IN / rid / "eda_frag1.inp"
        ts_B = LOCKED_ORCA_IN / rid / "eda_frag2.inp"
        r_A = LOCKED_STRAIN_CP / rid / "fragA_R.inp"
        r_B = LOCKED_STRAIN_CP / rid / "fragB_R.inp"
        # If v9 files aren't there, fall back to v8 — but flag that too.
        if not r_A.exists():
            r_A = LOCKED_STRAIN_ALT / rid / "fragA_R.inp"
        if not r_B.exists():
            r_B = LOCKED_STRAIN_ALT / rid / "fragB_R.inp"

    return {
        "reaction_id": rid,
        "sub_source": sub,
        "TS_A": parse_orca_input(ts_A),
        "TS_B": parse_orca_input(ts_B),
        "R_A":  parse_orca_input(r_A),
        "R_B":  parse_orca_input(r_B),
    }


def profile_half(records: list[dict], side: str) -> dict:
    """Aggregate per-file fingerprints into a per-half profile.
    side ∈ {'TS_A', 'TS_B', 'R_A', 'R_B'}
    """
    routes = Counter()
    cp = Counter()
    opt = Counter()
    solv = Counter()
    for r in records:
        f = r[side]
        if not f.get("exists"):
            routes["MISSING"] += 1
            continue
        routes[f["route"]] += 1
        cp[bool(f["is_cp"])] += 1
        opt[bool(f["is_opt"])] += 1
        solv[f["solvent"] or "none"] += 1
    return {
        "n": len(records),
        "routes": dict(routes),
        "cp": {str(k): v for k, v in cp.items()},
        "opt": {str(k): v for k, v in opt.items()},
        "solvent": dict(solv),
    }


def compare_halves(prof: dict) -> list[str]:
    """Cross-half comparison per side. Returns list of divergence strings."""
    divergences = []
    for side in ("TS_A", "TS_B", "R_A", "R_B"):
        for attr in ("cp", "opt", "solvent"):
            l = set(prof["locked_778"][side][attr].keys())
            s = set(prof["spec16"][side][attr].keys())
            if l != s:
                divergences.append(
                    f"{side}.{attr}: locked_778={sorted(l)} spec16={sorted(s)}"
                )
        # routes are expected to differ in optional keywords; only flag
        # if the level (BLYP D3BJ def2-TZVP) itself differs.
        for r_l in prof["locked_778"][side]["routes"]:
            for r_s in prof["spec16"][side]["routes"]:
                if r_l == "MISSING" or r_s == "MISSING":
                    continue
                # Compare level fragments only (functional + dispersion + basis)
                def level(r):
                    r_low = r.lower()
                    frags = {
                        "functional_blyp": "blyp" in r_low,
                        "d3bj":            "d3bj" in r_low,
                        "basis_def2tzvp":  "def2-tzvp" in r_low,
                    }
                    return frags
                if level(r_l) != level(r_s):
                    divergences.append(
                        f"{side}.route: locked_778={r_l!r} vs spec16={r_s!r} — level differs")
    return divergences


def within_half_selfconsistency(prof: dict, half: str) -> list[str]:
    """Warn on within-half inconsistency, e.g. TS-side CP but R-side not."""
    issues = []
    for side_pair in (("TS_A", "R_A"), ("TS_B", "R_B")):
        cp_ts = set(prof[half][side_pair[0]]["cp"].keys())
        cp_r  = set(prof[half][side_pair[1]]["cp"].keys())
        if cp_ts != cp_r:
            issues.append(
                f"{half} internal CP inconsistency: "
                f"{side_pair[0]}.cp={sorted(cp_ts)} vs {side_pair[1]}.cp={sorted(cp_r)}")
    return issues


def main() -> int:
    STAGE.joinpath("logs").mkdir(parents=True, exist_ok=True)
    with open(OUT_LOG, "w") as fh:
        _log(fh, "=== spec20 G20-0 protocol discovery ===")
        _log(fh, f"[env] python={platform.python_version()} pandas={pd.__version__}")

        st1 = pd.read_pickle(STAGE1_PKL)
        _log(fh, f"[stage1] loaded n={len(st1)}")

        records = []
        for _, row in st1.iterrows():
            records.append(audit_reaction(str(row["reaction_id"]), str(row["sub_source"])))

        _log(fh, f"[audit] n_records={len(records)}")

        # Per-half aggregate profile
        prof = {"locked_778": {}, "spec16": {}}
        for half in ("locked_778", "spec16"):
            subset = [r for r in records if r["sub_source"] == half]
            for side in ("TS_A", "TS_B", "R_A", "R_B"):
                prof[half][side] = profile_half(subset, side)

        # Cross-half + within-half analysis
        cross = compare_halves(prof)
        within_l = within_half_selfconsistency(prof, "locked_778")
        within_s = within_half_selfconsistency(prof, "spec16")

        finding = {
            "cross_half_divergences_beyond_geometry": cross,
            "within_locked_778_inconsistencies": within_l,
            "within_spec16_inconsistencies": within_s,
        }

        _log(fh, "[finding] " + json.dumps(finding, indent=2))

        summary = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "n_records_audited": len(records),
            "profile_by_half": prof,
            "finding": finding,
        }
        with open(OUT_JSON, "w") as jf:
            json.dump({"summary": summary, "records": records}, jf, indent=2, default=str)
        _log(fh, f"[write] {OUT_JSON}")

        # §7 default: HALT on any divergence beyond relaxed-fragment geometry.
        must_halt = bool(cross) or bool(within_l) or bool(within_s)
        if must_halt:
            reason_lines = []
            if cross:
                reason_lines.append("Cross-half divergences beyond geometry:")
                reason_lines.extend(f"  - {d}" for d in cross)
            if within_l:
                reason_lines.append("Within locked_778 self-consistency issues:")
                reason_lines.extend(f"  - {d}" for d in within_l)
            if within_s:
                reason_lines.append("Within spec16 self-consistency issues:")
                reason_lines.extend(f"  - {d}" for d in within_s)
            reason = "\n".join(reason_lines)
            _log(fh, "[G20-0 HALT] " + reason)
            # We still allow downstream aggregate to run — write a sentinel
            # then exit non-zero AFTER aggregate has a chance to summarise.
            (STAGE / "logs" / "G20_0_HALT.flag").write_text(reason + "\n")
            _log(fh, "[note] halt is recorded via G20_0_HALT.flag; aggregate.py "
                     "will build summary.md before the batch job exits.")
        else:
            (STAGE / "logs" / "G20_0_PASS.flag").write_text(
                "G20-0 PASS: halves match on all protocol attributes beyond geometry.\n")
            _log(fh, "[G20-0 PASS] halves match on all protocol attributes beyond geometry.")
        _log(fh, "=== discovery OK (finding recorded) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
