"""Check stratification of the 500 ASR-labeled reactions.

Parses reaction_id to extract source subset (T1x_ / Halogen_), molecular
formula, heavy-atom count, halogen content, and presence of sulfur.
Cross-tabulates the 500 reactions by these axes and flags cells whose
usable count falls below a threshold after EXCLUDED / FAIL reactions are
removed.

Outputs:
    <out_dir>/report.md             — human-readable report
    <out_dir>/crosstab__<cut>.csv   — one CSV per cross-tab
    stdout                          — short summary

Usage:
    python check_stratification.py --manifest manifest_v2.csv \\
        --out-dir ./strat_report [--thin-threshold 15]

    python check_stratification.py --selftest    # run built-in tests
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import pandas as pd  # pip install pandas


HEAVY_ELEMENTS = frozenset({"C", "N", "O", "F", "P", "S", "Cl", "Br", "I"})
HALOGENS = frozenset({"F", "Cl", "Br", "I"})
USABLE_VERDICTS = frozenset({"PASS", "WARN", "PENDING_REVALIDATE"})
DROPPED_VERDICTS = frozenset({"EXCLUDED", "FAIL"})


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReactionMeta:
    reaction_id: str
    source_subset: str       # 'T1x' or 'Halogen'
    formula: str             # e.g. 'BrC4H4NO'
    heavy_atom_count: int
    halogen_label: str       # 'none', 'F', 'Cl', 'Br', or 'multi'
    has_sulfur: bool
    has_nitrogen: bool
    verdict: str             # uppercased


def parse_reaction_id(rid: str) -> tuple[str, str]:
    """Return (source_subset, formula) parsed from reaction_id.

    Expects 'T1x_<formula>_rxn<digits>' or 'Halogen_<formula>_rxn<digits>'.
    """
    parts = rid.split("_")
    if len(parts) < 3 or not parts[-1].startswith("rxn"):
        raise ValueError(f"unexpected reaction_id format: {rid}")
    source = parts[0]
    formula = "_".join(parts[1:-1])
    if source not in ("T1x", "Halogen"):
        raise ValueError(f"unknown source subset {source!r} in {rid}")
    return source, formula


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a Hill-like molecular formula into element -> count.

    Element token: one uppercase letter optionally followed by one lowercase
    (e.g. C, Cl, Br). Count suffix: optional digit run; absent means 1.
    """
    counts: dict[str, int] = {}
    i, n = 0, len(formula)
    while i < n:
        if not formula[i].isupper():
            raise ValueError(f"unexpected char {formula[i]!r} at pos {i} in {formula!r}")
        if i + 1 < n and formula[i + 1].islower():
            elem, i = formula[i:i + 2], i + 2
        else:
            elem, i = formula[i], i + 1
        j = i
        while j < n and formula[j].isdigit():
            j += 1
        count = int(formula[i:j]) if j > i else 1
        i = j
        if elem != "H" and elem not in HEAVY_ELEMENTS:
            raise ValueError(f"unknown element {elem!r} in {formula!r}")
        counts[elem] = counts.get(elem, 0) + count
    return counts


def classify(rid: str, verdict: str) -> ReactionMeta:
    source, formula = parse_reaction_id(rid)
    elem_counts = parse_formula(formula)
    heavy = sum(c for e, c in elem_counts.items() if e in HEAVY_ELEMENTS)
    halogens = sorted(e for e in elem_counts if e in HALOGENS)
    if not halogens:
        halogen_label = "none"
    elif len(halogens) == 1:
        halogen_label = halogens[0]
    else:
        halogen_label = "multi"
    return ReactionMeta(
        reaction_id=rid,
        source_subset=source,
        formula=formula,
        heavy_atom_count=heavy,
        halogen_label=halogen_label,
        has_sulfur="S" in elem_counts,
        has_nitrogen="N" in elem_counts,
        verdict=verdict.strip().upper(),
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> tuple[list[ReactionMeta], list[tuple[str, str]]]:
    """Return (classified, errors) where errors is [(rid, reason), ...].

    Prefers `new_verdict` column (fix_fail_19 output) over `verdict`.
    """
    classified: list[ReactionMeta] = []
    errors: list[tuple[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "reaction_id" not in fields:
            raise SystemExit("manifest missing 'reaction_id' column")
        if "new_verdict" not in fields and "verdict" not in fields:
            raise SystemExit("manifest needs either 'new_verdict' or 'verdict' column")
        for row in reader:
            rid = row["reaction_id"]
            v_new = (row.get("new_verdict") or "").strip()
            v_old = (row.get("verdict") or "").strip()
            verdict = v_new if v_new else v_old
            try:
                classified.append(classify(rid, verdict))
            except ValueError as e:
                errors.append((rid, str(e)))
    return classified, errors


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def to_dataframe(metas: list[ReactionMeta]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "reaction_id": m.reaction_id,
            "source_subset": m.source_subset,
            "formula": m.formula,
            "heavy_atom_count": m.heavy_atom_count,
            "halogen_label": m.halogen_label,
            "has_sulfur": m.has_sulfur,
            "has_nitrogen": m.has_nitrogen,
            "verdict": m.verdict,
            "is_usable": m.verdict in USABLE_VERDICTS,
            "is_dropped": m.verdict in DROPPED_VERDICTS,
        }
        for m in metas
    ])


def crosstab(df: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    """Per-cell counts of total / usable / dropped, sorted by usable asc."""
    grouped = df.groupby(axes, dropna=False).agg(
        total=("reaction_id", "size"),
        usable=("is_usable", "sum"),
        dropped=("is_dropped", "sum"),
    ).reset_index()
    return grouped.sort_values("usable").reset_index(drop=True)


def df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table (no deps)."""
    if df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

CUTS: list[tuple[str, list[str]]] = [
    ("source_subset (overall)",                    ["source_subset"]),
    ("source_subset × heavy_atom_count",           ["source_subset", "heavy_atom_count"]),
    ("source_subset × halogen_label",              ["source_subset", "halogen_label"]),
    ("source_subset × has_sulfur",                 ["source_subset", "has_sulfur"]),
    ("source × heavy × halogen (finest)",          ["source_subset", "heavy_atom_count", "halogen_label"]),
    ("source × heavy × has_sulfur (S-risk grid)",  ["source_subset", "heavy_atom_count", "has_sulfur"]),
]


def write_report(metas: list[ReactionMeta], errors: list[tuple[str, str]],
                 out_dir: str, thin_threshold: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    df = to_dataframe(metas)

    n_total = len(df)
    n_usable = int(df["is_usable"].sum())
    n_dropped = int(df["is_dropped"].sum())
    n_other = n_total - n_usable - n_dropped

    md: list[str] = [
        "# Stratification report\n",
        f"- Total reactions classified: **{n_total}**",
        f"- Usable (PASS / WARN / PENDING_REVALIDATE): **{n_usable}**",
        f"- Dropped (EXCLUDED / FAIL): **{n_dropped}**",
    ]
    if n_other:
        md.append(f"- Other verdict (uncategorized): **{n_other}**")
    md.append(f"- Thin-cell threshold: usable < **{thin_threshold}**\n")

    if errors:
        md.append(f"## ⚠ Parsing errors: {len(errors)}\n")
        for rid, err in errors[:20]:
            md.append(f"- `{rid}`: {err}")
        if len(errors) > 20:
            md.append(f"- ... and {len(errors) - 20} more")
        md.append("")

    # All cross-tabs
    thin_summary: list[tuple[str, list[str], pd.DataFrame]] = []
    for title, axes in CUTS:
        ct = crosstab(df, axes)
        md.append(f"\n## Cross-tab — {title}\n")
        md.append(df_to_md(ct))
        thin = ct[ct["usable"] < thin_threshold]
        if len(thin):
            md.append(f"\n**⚠ {len(thin)} thin cell(s) (usable < {thin_threshold}):**\n")
            for _, row in thin.iterrows():
                cell_id = ", ".join(f"{a}={row[a]}" for a in axes)
                md.append(f"- {cell_id} → usable={row['usable']}, dropped={row['dropped']}")
            thin_summary.append((title, axes, thin))
        ct.to_csv(os.path.join(out_dir, "crosstab__" + "_".join(axes) + ".csv"),
                  index=False)

    # Dropped reactions detail
    dropped_df = df[df["is_dropped"]].sort_values(
        ["source_subset", "heavy_atom_count", "has_sulfur"]
    )
    if len(dropped_df):
        md.append("\n## Dropped reactions — provenance\n")
        keep_cols = ["reaction_id", "source_subset", "heavy_atom_count",
                     "halogen_label", "has_sulfur", "verdict"]
        md.append(df_to_md(dropped_df[keep_cols].reset_index(drop=True)))

    # Action items: collapse thin findings into a clear decision
    md.append("\n## Action items\n")
    if thin_summary:
        md.append(f"Cells with **usable < {thin_threshold}** found in "
                  f"{len(thin_summary)} cut(s). Replacement candidates "
                  "should target these strata (T1x_ subset preferred, "
                  "and run cheap endpoint pre-check before D2AF + ADF EDA):\n")
        seen: set[tuple] = set()
        for title, axes, thin in thin_summary:
            for _, row in thin.iterrows():
                key = tuple((a, row[a]) for a in axes)
                if key in seen:
                    continue
                seen.add(key)
                cell_id = ", ".join(f"**{a}={row[a]}**" for a in axes)
                md.append(f"- {cell_id} → usable={row['usable']}")
    else:
        md.append(f"No cell falls below threshold {thin_threshold}. "
                  f"**No replacement needed** — the {n_usable} usable "
                  "reactions cover all strata adequately.")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md) + "\n")
    return report_path


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

def selftest() -> int:
    """Verify parsing against known reaction IDs from this project."""
    cases = [
        # (rid, source, heavy, halogen_label, has_sulfur, has_nitrogen)
        ("Halogen_BrC4H4NS_rxn10113", "Halogen", 7, "Br", True,  True),
        ("Halogen_C4ClH5N2_rxn12941", "Halogen", 7, "Cl", False, True),
        ("Halogen_C5FH5S_rxn16443",   "Halogen", 7, "F",  True,  False),
        ("Halogen_C4ClH4NS_rxn12917", "Halogen", 7, "Cl", True,  True),
        ("Halogen_BrC4H4NO_rxn10056", "Halogen", 7, "Br", False, True),
        ("Halogen_C4ClH4NS_rxn12932", "Halogen", 7, "Cl", True,  True),
        ("Halogen_C4FH5N2O_rxn14222", "Halogen", 8, "F",  False, True),
        ("T1x_C5H6O_rxn06161",        "T1x",     6, "none", False, False),
        ("T1x_C7H12_rxn09748",        "T1x",     7, "none", False, False),
        ("T1x_C5H9NO_rxn08047",       "T1x",     7, "none", False, True),
        ("Halogen_BrC6H4N_rxn11968",  "Halogen", 8, "Br", False, True),
    ]
    fails = []
    for rid, exp_src, exp_heavy, exp_hal, exp_S, exp_N in cases:
        try:
            m = classify(rid, "PASS")
        except ValueError as e:
            fails.append((rid, f"raised: {e}"))
            continue
        actual = (m.source_subset, m.heavy_atom_count, m.halogen_label,
                  m.has_sulfur, m.has_nitrogen)
        expected = (exp_src, exp_heavy, exp_hal, exp_S, exp_N)
        if actual != expected:
            fails.append((rid, f"expected {expected}, got {actual}"))
    if fails:
        for rid, reason in fails:
            print(f"FAIL {rid}: {reason}", file=sys.stderr)
        return 1
    print(f"selftest: {len(cases)}/{len(cases)} OK")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", help="manifest_v2.csv (or original manifest.csv)")
    p.add_argument("--out-dir", default="./strat_report",
                   help="output directory for report.md and crosstab CSVs")
    p.add_argument("--thin-threshold", type=int, default=15,
                   help="cells with usable < this are flagged (default 15)")
    p.add_argument("--selftest", action="store_true",
                   help="run built-in parser tests and exit")
    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if not args.manifest:
        p.error("--manifest is required (unless --selftest)")
    if not os.path.exists(args.manifest):
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(2)

    metas, errors = load_manifest(args.manifest)
    if not metas and not errors:
        print("no rows in manifest", file=sys.stderr)
        sys.exit(2)
    if errors and not metas:
        print(f"all {len(errors)} rows failed to parse", file=sys.stderr)
        sys.exit(2)

    report_path = write_report(metas, errors, args.out_dir, args.thin_threshold)

    n_total = len(metas)
    n_usable = sum(1 for m in metas if m.verdict in USABLE_VERDICTS)
    n_dropped = sum(1 for m in metas if m.verdict in DROPPED_VERDICTS)
    print(f"classified: {n_total}  (usable {n_usable}, dropped {n_dropped})")
    if errors:
        print(f"parsing errors: {len(errors)} (see {report_path})")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
