#!/usr/bin/env python3
"""tests/ext_probe.py — probe of ORCA 6.1.1 `! ExtOpt` (start condition 0-5 / A-1 groundwork).

Runs ORCA on water with `! ExtOpt EnGrad` and `%method ProgExt "<probe.sh>" Ext_Params "--probe" end`; the external
program is a bash script that logs its argv, the working directory listing and every file ORCA passed, then writes a
harmonic-model energy + gradient in the .engrad format next to the input file ORCA named. Everything ORCA printed is
kept in $MLT/tmp/extprobe/ for reading. Run inside the sandbox:  bash jobs/sbx.sh reactot tests/ext_probe.py"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlt_common as C  # noqa: E402

D = C.assert_write_ok(C.MLT / "tmp" / "extprobe")
if D.exists():
    shutil.rmtree(D)
D.mkdir(parents=True)
probe = D / "probe.sh"
probe.write_text(r"""#!/bin/bash
LOG=__D__/probe_calls.log
{
echo "=== call $(date +%T.%N) pwd=$(pwd)"
echo "argv: $*"
ls -la
for f in "$@"; do if [ -f "$f" ]; then echo "--- file $f"; cat "$f"; fi; done
} >> $LOG 2>&1
inp=$1
base=${inp%.extinp.tmp}
xyz=$(sed -n 1p "$inp" | awk '{print $1}')
n=$(head -1 "$xyz" | awk '{print $1}')
echo "--- xyz $xyz"; cat "$xyz" >> $LOG
{
echo "#"; echo "# Number of atoms"; echo "#"; echo " $n"
echo "#"; echo "# The current total energy in Eh"; echo "#"; echo "   -76.000000000000"
echo "#"; echo "# The current gradient in Eh/bohr"; echo "#"
for i in $(seq 1 $((3*n))); do echo "       0.000000000000"; done
} > "${base}.engrad"
echo "wrote ${base}.engrad" >> $LOG
""".replace("__D__", str(D)))
probe.chmod(0o755)
inp = D / "w.inp"
inp.write_text(f"""! ExtOpt EnGrad
%method
  ProgExt "{probe}"
  Ext_Params "--probe"
end
*xyz 0 1
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
*
""")
r = subprocess.run([str(C.ORCA), "w.inp"], cwd=D, capture_output=True, text=True, timeout=300)
(D / "w.out").write_text(r.stdout)
(D / "w.err").write_text(r.stderr)
print("rc", r.returncode)
print("--- files", sorted(p.name for p in D.iterdir()))
print("--- probe log"); print((D / "probe_calls.log").read_text()[:4000] if (D / "probe_calls.log").exists() else "(no call)")
print("--- out tail"); print("\n".join(r.stdout.splitlines()[-40:]))
print("--- err"); print(r.stderr[-2000:])
