#!/usr/bin/env python3
"""Pass/fail gate for the SMD relabel, before and after building labels_all.json.

  smd_relabel_gate.py pre          audit/audit_full.csv + audit/uniformity.tsv + state
                                   -> PASS only if every criterion below holds
  smd_relabel_gate.py post LABELS  LABELS (a built labels_all.json) must reproduce
                                   the audit values for every rxn

Label convention (option ①, 2026-09-28): d1/d2/eint_spe from the standalone
frag{1,2}_dist.out; ORCA's Bond Energy / 6 channels refer to eda_frag*.out
(ghost basis) and the difference is the bsse_shift column.

pre criteria:
  completeness  every label_true rxn is local and has a done flag; no live claim
  audit         one row per rxn, gates computable, zero file/input/level problems
                (incl. frag*_dist: terminated, 1 FSPE, SMD, TS coordinates)
  gate 1        |d1 + d2 + eint_spe - barrier| < 1e-4 kcal/mol for all
  gate 2        |Bond Energy - (E(AB) - E(eda_frag1) - E(eda_frag2))| < 0.01,
                i.e. eint_spe = e_bond + bsse_shift with bsse_shift taken from
                the same eda_frag files ORCA used
  gate 3        |d1 + d2 + sum(6ch) + bsse_shift - barrier| < 0.03 kcal/mol, or
                the miss is exactly ORCA's own EDA-table closure (rows - Bond
                Energy) and below 0.15 kcal/mol (the known COSX/grid residual)
  SMD uniform   every uniformity.tsv column holds a single value over all rxns,
                and every output used (eda, eda_frag1/2, frag1/2_dist,
                frag1/2_rel) is ORCA 6.1.1, SMD module, SMD CDS term, water,
                eps 78.3550, terminated normally

Results are appended to audit/GATE_REPORT.txt. Exit code 0 = PASS.
"""
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

EH = 627.5094740631
ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel")
AUDIT = ROOT / "audit"
OLD = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/inputs")
REPORT = Path(os.environ.get("GATE_REPORT", str(AUDIT / "GATE_REPORT.txt")))
TOL3, TOL3_CLOSURE = 0.03, 0.15
EXCLUDED = {3090, 3766, 4252, 3400, 5783}
RXN = re.compile(r"rxn_\d+")
OUTS = ("eda", "ef1", "ef2", "sd1", "sd2", "rel1", "rel2")


class Gate:
    def __init__(self, title):
        self.lines, self.ok = [f"=== {title}  {datetime.now().isoformat(timespec='seconds')}"], True

    def check(self, cond, msg):
        self.lines.append(("PASS  " if cond else "FAIL  ") + msg)
        self.ok = self.ok and bool(cond)

    def info(self, msg):
        self.lines.append("INFO  " + msg)

    def finish(self):
        self.lines.append(f"=== RESULT: {'PASS' if self.ok else 'FAIL'}\n")
        text = "\n".join(self.lines)
        with open(REPORT, "a") as fh:
            fh.write(text + "\n")
        print(text)
        return 0 if self.ok else 1


def pre():
    g = Gate("pre-build gate (option ①)")
    all_ids = sorted(p.name for p in OLD.iterdir() if RXN.fullmatch(p.name))
    local = {p.name for p in (ROOT / "inputs").iterdir() if RXN.fullmatch(p.name)}
    done = {p.name for p in (ROOT / "state" / "done").iterdir()}
    claims = sorted(p.name for p in (ROOT / "state" / "claim").iterdir() if RXN.fullmatch(p.name))
    g.check(len(all_ids) == 5265, f"label_true rxns = {len(all_ids)} (expect 5265)")
    g.check(set(all_ids) <= local, f"all rxns local (outstanding = {len(set(all_ids) - local)})")
    g.check(set(all_ids) <= done, f"all rxns done (not done = {sorted(set(all_ids) - done)[:10]})")
    g.check(not claims, f"no live claims ({claims[:10]})")

    rows = list(csv.DictReader(open(AUDIT / "audit_full.csv")))
    g.check(len(rows) == len(all_ids) and {r["rxn"] for r in rows} == set(all_ids),
            f"audit rows = {len(rows)} cover every rxn")
    g.check(all(r["gates_computable"] == "True" and r["d1_std"] != "" for r in rows),
            "gates computable for all (incl. frag*_dist energies)")
    bad = [r["rxn"] for r in rows if r["problems"]]
    probs = Counter(p for r in rows for p in r["problems"].split(";") if p)
    g.check(not bad, f"file/input/level problems = {len(bad)} {dict(probs.most_common(8))} {bad[:10]}")
    m1 = [r["rxn"] for r in rows if abs(float(r["asm_res_std"])) >= 1e-4]
    g.check(not m1, f"gate 1 d1+d2+eint_spe = barrier: {len(rows) - len(m1)}/{len(rows)} {m1[:10]}")
    m2 = [r["rxn"] for r in rows if abs(float(r["bsse_res"])) >= 0.01]
    g.check(not m2, f"gate 2 Bond = E(AB)-E(eda_frag1)-E(eda_frag2) (eint_spe = e_bond + bsse_shift): "
                    f"{len(rows) - len(m2)}/{len(rows)} {m2[:10]}")
    closure, g3bad = [], []
    for r in rows:
        e, c = float(r["closure_std"]), float(r["rows_minus_bond_eh"]) * EH
        if abs(e) < TOL3:
            continue
        (closure if abs(e - c) < 1e-6 and abs(e) < TOL3_CLOSURE else g3bad).append(r["rxn"])
    g.check(not g3bad, f"gate 3 d1+d2+6ch+bsse_shift = barrier: misses not explained by ORCA table closure = "
                       f"{len(g3bad)} {g3bad[:10]}")
    g.info(f"gate 3 within {TOL3}: {len(rows) - len(closure) - len(g3bad)}; ORCA table-closure residual "
           f"({TOL3}-{TOL3_CLOSURE} kcal/mol): {len(closure)}; max |residual| = "
           f"{max(abs(float(r['closure_std'])) for r in rows):.4f}")
    g.info(f"stage3-equivalent status: {dict(Counter(r['stage3_status'] for r in rows))}")

    uni = list(csv.DictReader(open(AUDIT / "uniformity.tsv"), delimiter="\t"))
    g.check({u["rxn"] for u in uni} == set(all_ids), f"uniformity scan covers every rxn ({len(uni)})")
    for col in uni[0].keys():
        if col == "rxn":
            continue
        vals = Counter(u[col] for u in uni)
        g.check(len(vals) == 1, f"uniform {col}: {dict(vals.most_common(3))}")
    for k in OUTS:
        v = {c: Counter(u[f"{k}_{c}"] for u in uni) for c in ("version", "solvent", "eps", "smdmod", "cds", "term")}
        ok = (all(x.startswith("Program Version 6.1.1") for x in v["version"])
              and set(v["solvent"]) == {"WATER"} and set(v["eps"]) == {"78.3550"}
              and all(x.isdigit() and int(x) >= 1 for c in ("smdmod", "cds", "term") for x in v[c]))
        g.check(ok, f"{k}.out is ORCA 6.1.1 / SMD / CDS / water / eps 78.3550 / terminated: "
                    f"{ {c: dict(x.most_common(2)) for c, x in v.items()} }")
    return g.finish()


def post(labels_path):
    g = Gate(f"post-build crosscheck (option ①) {labels_path}")
    data = json.load(open(labels_path))
    rows = {r["rxn"]: r for r in csv.DictReader(open(AUDIT / "audit_full.csv"))}
    g.check(len(data) == len(rows), f"records = {len(data)} (audit rows {len(rows)})")
    ids = {f"rxn_{int(d['rxn_id']):04d}" for d in data}
    g.check(ids == set(rows), f"record ids == audit ids (missing {sorted(set(rows) - ids)[:10]})")
    pairs = [("d1_kcal", "d1_std"), ("d2_kcal", "d2_std"), ("eint_spe_kcal", "eint_spe_std"),
             ("barrier_kcal", "barrier"), ("e_bond_kcal", "e_bond"), ("bsse_shift_kcal", "bsse_shift"),
             ("elst_dft", "elst"), ("pauli_dft", "pauli_dft"), ("oi_dft", "oi"), ("disp_dft", "disp"),
             ("cpcm_dft", "cpcm"), ("cds_dft", "cds"), ("e_ab_eh", "E_eda"),
             ("e_frag1_dist_eh", "E_sd1"), ("e_frag2_dist_eh", "E_sd2"),
             ("e_frag1_dist_ghost_eh", "E_ef1"), ("e_frag2_dist_ghost_eh", "E_ef2"),
             ("e_frag1_rel_eh", "E_rel1"), ("e_frag2_rel_eh", "E_rel2")]
    worst, ident = {}, Counter()
    for d in data:
        r = rows.get(f"rxn_{int(d['rxn_id']):04d}")
        if r is None:
            continue
        for a, b in pairs:
            diff = abs(float(d[a]) - float(r[b]))
            if diff > worst.get(a, (0, ""))[0]:
                worst[a] = (diff, r["rxn"])
        if abs(d["identity_residual_kcal"]) >= 1e-6:
            ident["d1+d2+eint_spe!=barrier"] += 1
        if abs(d["d1_kcal"] + d["d2_kcal"] + d["e_bond_kcal"] + d["bsse_shift_kcal"] - d["barrier_kcal"]) >= 1e-6:
            ident["d1+d2+e_bond+bsse_shift!=barrier"] += 1
        if abs(d["bsse_shift_kcal"] - d["bsse_shift1_kcal"] - d["bsse_shift2_kcal"]) >= 0.01:
            ident["bsse_shift!=bsse1+bsse2"] += 1
        if abs(d["bond_reference_residual_kcal"]) >= 0.01:
            ident["bond_reference"] += 1
        if (abs(d["channel_closure_residual_kcal"] - d["eda_sum_minus_bond_kcal"]) >= 1e-6
                or abs(d["channel_closure_residual_kcal"]) >= TOL3_CLOSURE):
            ident["channel_closure"] += 1
        if "e_frag1_dist_standalone_eh" in d or d["e_frag1_dist_eh"] == d["e_frag1_dist_ghost_eh"]:
            ident["not_option1"] += 1
    mx = max((v[0] for v in worst.values()), default=0.0)
    g.check(mx < 1e-8, f"labels reproduce audit values (max |diff| = {mx:.2e}); worst: "
                       f"{ {k: f'{v[0]:.1e}@{v[1]}' for k, v in worst.items() if v[0] > 0} }")
    g.check(not ident, f"per-record identities (option ①) violated: {dict(ident)}")
    st = Counter(d["status"] for d in data)
    g.check(set(st) <= {"ok", "excluded"}, f"status counts {dict(st)}")
    exc = {int(d["rxn_id"]) for d in data if d["status"] == "excluded"}
    g.check(exc == EXCLUDED, f"excluded = {sorted(exc)}")
    g.info(f"promoted eda_sum_mismatch -> ok: {sum(1 for d in data if d.get('sum_mismatch_promoted'))}; "
           f"exclusion reasons: { {int(d['rxn_id']): d.get('exclusion_reason') for d in data if d['status'] == 'excluded'} }")
    return g.finish()


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "pre":
        sys.exit(pre())
    if len(sys.argv) >= 3 and sys.argv[1] == "post":
        sys.exit(post(sys.argv[2]))
    sys.exit("usage: smd_relabel_gate.py pre | post LABELS")
