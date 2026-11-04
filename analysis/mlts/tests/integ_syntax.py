#!/usr/bin/env python3
"""tests/integ_syntax.py — Integrate step: compile() every analysis/mlts module (no .pyc written) and print the
errors. Compute nodes only: bash jobs/sbx.sh reactot tests/integ_syntax.py [files...]"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
files = sys.argv[1:] or sorted(str(p.name) for p in HERE.glob("*.py")) + [f"tests/{p.name}" for p in
                                                                         sorted((HERE / "tests").glob("*.py"))]
bad = 0
for f in files:
    p = HERE / f
    try:
        compile(p.read_text(), str(p), "exec", dont_inherit=True)
    except SyntaxError as e:
        bad += 1
        print(f"SYNTAX {f}:{e.lineno}: {e.msg}\n    {(e.text or '').rstrip()}")
print(f"{len(files)} files, {bad} with syntax errors")
sys.exit(1 if bad else 0)
