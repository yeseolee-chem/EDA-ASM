#!/usr/bin/env python3
"""S0: what the smoke job (HCNO + ethylene, full chain) must show before any D1 job runs.
  - autodE wrote ORCA inputs at the intended level (B3LYP D3BJ def2-SVP, SMD water, OptTS block)
  - autodE parsed ORCA 6.1.1 outputs: a TS with exactly one imaginary frequency
  - the chain reached a label that passes every gate
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_common as pc

cfg = pc.load_config()
jd = Path(cfg["scratch"]) / "jobs" / "S0"
res = []
def chk(name, ok, detail=""):
    res.append((name, bool(ok), detail)); print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

inps = list((jd / "ts").rglob("*_orca.inp"))
optts = [p for p in inps if "optts" in p.name]
sp = [p for p in inps if p.name.endswith("_sp_orca.inp")]
t = optts[0].read_text() if optts else ""
chk("autodE wrote ORCA OptTS inputs", optts, f"{len(inps)} ORCA inputs, {len(optts)} optts")
for kw in ("OptTS", "Freq", "B3LYP", "D3BJ", "def2-SVP", "CPCM(Water)", "smd true", 'SMDsolvent "water"',
           "Recalc_Hess 30", "Trust -0.1", "MaxIter 100"):
    chk(f"optts input contains {kw!r}", kw in t)
chk("sp inputs use def2-TZVP", sp and all("def2-TZVP" in p.read_text() for p in sp), f"{len(sp)} sp inputs")
outs = list((jd / "ts").rglob("*_orca.out"))
vers = {m.group(1) for p in outs for m in [re.search(r"Program Version\s+(\S+)", p.read_text(errors="replace"))] if m}
chk("all autodE ORCA outputs are 6.1.1", vers == {"6.1.1"}, str(vers))
ts = json.loads((jd / "ts_result.json").read_text()) if (jd / "ts_result.json").is_file() else {}
chk("TS found and parsed (one imaginary frequency)", ts.get("n_imag") == 1, str(ts.get("imag_freqs_cm")))
chk("imaginary mode on the forming bonds", (ts.get("checks") or {}).get("imag_mode_on_forming_bonds"),
    str(ts.get("imag_mode_forming_share")))
lab = json.loads((jd / "label.json").read_text()) if (jd / "label.json").is_file() else {}
chk("smoke label ok with all gates", pc.done(jd, "label") and all((lab.get("gates") or {"x": False}).values()),
    str(lab.get("gates")))
bad = [r for r in res if not r[1]]
print("SMOKE", "PASS" if not bad else f"FAIL ({len(bad)})")
sys.exit(1 if bad else 0)
