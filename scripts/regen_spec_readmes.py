"""Regenerate spec/*/README.md based on current directory state.

Introspects each spec folder and composes a README with:
  * folder name → title
  * primary code file docstring (from the largest .py in code/)
  * files & sizes summary (code/, results/, figures/, oof/, splits/, data/)
  * summary from results/summary.md or REPORT.md if present
  * status (has results/figures? empty? etc.)

Idempotent: always regenerates.
"""
from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SPEC_ROOT = REPO / "spec"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".{}_".format(path.name), dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def get_docstring(pyfile: Path) -> str:
    try:
        tree = ast.parse(pyfile.read_text(errors="ignore"))
        return ast.get_docstring(tree) or ""
    except Exception:
        return ""


def _summarize_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def list_files(subdir: Path, max_show: int = 30) -> list[tuple[str, int]]:
    if not subdir.is_dir():
        return []
    entries = []
    for p in sorted(subdir.iterdir()):
        if p.is_file():
            try:
                entries.append((p.name, p.stat().st_size))
            except OSError:
                pass
    return entries[:max_show]


def find_primary_py(code_dir: Path) -> Path | None:
    if not code_dir.is_dir():
        return None
    pys = [p for p in code_dir.iterdir() if p.is_file() and p.suffix == ".py"
           and not p.name.startswith("_")]
    if not pys:
        return None
    # prioritise scripts named run_*, main*, or the largest by size
    prio = [p for p in pys if p.name.startswith(("run_", "main"))]
    if prio:
        return max(prio, key=lambda p: p.stat().st_size)
    return max(pys, key=lambda p: p.stat().st_size)


def find_summary_md(res_dir: Path) -> Path | None:
    if not res_dir.is_dir():
        return None
    candidates = ["summary.md", "REPORT.md", "report.md", "lc_report.md",
                  "ood_report.md", "gate_verdict.md"]
    for name in candidates:
        p = res_dir / name
        if p.exists():
            return p
    return None


def compose_readme(spec_dir: Path) -> str:
    name = spec_dir.name
    display = name.replace("_", " ")
    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")

    # docstring from primary script
    code = spec_dir / "code"
    docstring = ""
    primary_py = find_primary_py(code)
    if primary_py is not None:
        docstring = get_docstring(primary_py)
    if docstring:
        # take only the first paragraph (up to first double newline) for the header
        head = docstring.strip().split("\n\n")[0].strip()
        lines.append(head)
        lines.append("")
    else:
        lines.append(f"*({display})*")
        lines.append("")

    # summary from results/*.md if present
    summary_md = find_summary_md(spec_dir / "results")
    if summary_md is not None:
        text = summary_md.read_text(errors="ignore").strip()
        # embed first 40 lines of the summary if it looks structured
        # (skip the top-level heading since we have our own)
        text_lines = text.splitlines()
        # drop leading H1
        while text_lines and text_lines[0].startswith("# "):
            text_lines = text_lines[1:]
        while text_lines and not text_lines[0].strip():
            text_lines = text_lines[1:]
        preview = "\n".join(text_lines[:60])
        lines.append(f"## Summary")
        lines.append("")
        lines.append(f"*(auto-embedded from `results/{summary_md.name}`)*")
        lines.append("")
        lines.append(preview)
        lines.append("")

    # Directory contents
    lines.append("## Directory contents")
    lines.append("")
    subdirs = ["code", "data", "splits", "oof", "results", "figures", "logs"]
    for sub in subdirs:
        sd = spec_dir / sub
        if not sd.is_dir():
            continue
        files = list_files(sd, max_show=30)
        if not files:
            lines.append(f"- `{sub}/` — (empty)")
            continue
        total_bytes = sum(sz for _, sz in files)
        # count files (may be more than max_show)
        all_files = [p for p in sd.iterdir() if p.is_file()]
        n_total = len(all_files)
        lines.append(f"- `{sub}/` — {n_total} file{'s' if n_total != 1 else ''} "
                     f"({_summarize_bytes(sum(p.stat().st_size for p in all_files if p.is_file()))})")
        for fname, sz in files:
            lines.append(f"    - `{fname}` ({_summarize_bytes(sz)})")
        if n_total > len(files):
            lines.append(f"    - … {n_total - len(files)} more")
    lines.append("")

    # Primary script docstring (if longer than the head we already used)
    if docstring and len(docstring) > 200:
        remainder = docstring.strip().split("\n\n", 1)
        if len(remainder) > 1 and remainder[1].strip():
            lines.append("## Primary script docstring")
            lines.append("")
            lines.append(f"`code/{primary_py.name}`:")
            lines.append("")
            lines.append("```")
            lines.append(docstring.strip())
            lines.append("```")
            lines.append("")

    lines.append("---")
    lines.append(f"*Auto-generated on {os.popen('date -Is').read().strip()} "
                 f"by `scripts/regen_spec_readmes.py`.*")
    lines.append("")
    return "\n".join(lines)


def main():
    if not SPEC_ROOT.is_dir():
        raise SystemExit(f"spec/ not found at {SPEC_ROOT}")

    specs = sorted([p for p in SPEC_ROOT.iterdir() if p.is_dir() and p.name.startswith("spec")])
    print(f"regenerating READMEs for {len(specs)} spec folders")

    for sd in specs:
        readme = sd / "README.md"
        # skip any README that starts with the manual marker
        if readme.exists():
            first_line = readme.read_text(errors="ignore").splitlines()[:1]
            if first_line and "manual: do-not-regen" in first_line[0]:
                print(f"  [skip-manual] {readme}")
                continue
        text = compose_readme(sd)
        atomic_write_text(readme, text)
        print(f"  [wrote] {readme}  ({len(text)} chars)")

    print(f"\ndone: {len(specs)} READMEs regenerated")


if __name__ == "__main__":
    main()
