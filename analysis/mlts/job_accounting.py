#!/usr/bin/env python
"""One-time job accounting (user request 11-05): sacct for yeseo1ee since 2026-11-04T00:00,
allocations only (-X) with array elements expanded, classified by job name into
본 계산 / 검사 · 시험 · dry-run · 시뮬레이터 / launcher / unclassified.
Writes results/JOB_ACCOUNTING.md (short). Run via jobs/mlt.slurm on a compute node:
  sbatch -p cpu1,cpu2 -c 1 --mem=2G -J mlt_acct --time=48:00:00 --export=ALL,MLTENV=reactot \
    jobs/mlt.slurm job_accounting.py
"""
import collections
import datetime as dt
import fnmatch
import os
import re
import subprocess
import sys

USER = "yeseo1ee"
START = "2026-11-04T00:00"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "JOB_ACCOUNTING.md")
FIELDS = ["JobID", "JobName", "Partition", "State", "Elapsed", "ElapsedRaw", "AllocCPUS",
          "AllocTRES", "CPUTimeRAW", "Start", "End"]

MAIN, CHECK, LAUNCH, UNK = "본 계산", "검사 · 시험 · dry-run · 시뮬레이터", "launcher", "unclassified"

# order matters: launcher -> checks -> main; anything else is unclassified (listed for the lead)
LAUNCH_PAT = ["mltp_launch"]
CHECK_PAT = ["mlt_simtest*", "mlt_test*", "mltp_test*", "*smoke*", "*sandbox*", "mlt_status*",
             "mlt_quota*", "mlt_audit*", "refcheck*", "refB*", "mlt_cancel*", "mlt_refcopy*",
             "verify*", "*dry*", "*simul*", "mlt_acct*"]
MAIN_PAT = ["g_run*", "g_collect*", "g_plan*", "g_g0*", "d0_*", "g_d0*", "nur_*", "nurion*",
            "ft_data*", "ft_p_*", "ft_spec_*", "ft_abl_*", "ftall_now_*", "mltp_FT-*", "mltp_G4*",
            "mltp_FT2-*", "ft2_spec*", "ft2_abl*", "z_*", "app_pilot*", "ftfull_*", "mltp_*",
            "mlt_g4*"]


def classify(name):
    for cls, pats in ((LAUNCH, LAUNCH_PAT), (CHECK, CHECK_PAT), (MAIN, MAIN_PAT)):
        if any(fnmatch.fnmatchcase(name, p) for p in pats):
            return cls
    return UNK


def prefix(name):
    """name prefix for subtotals: drop trailing fold/seed/shard digits, keep the stem."""
    p = re.sub(r"(_\d+)+$", "", name)
    return p or name


def gpus(tres):
    m = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", tres or "")
    return int(m.group(1)) if m else 0


def to_int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def main():
    cmd = ["sacct", "-u", USER, "-S", START, "-E", "now", "-X", "--array", "-P", "-n",
           "--format=" + ",".join(FIELDS)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"sacct failed rc={r.returncode}: {r.stderr.strip()[:500]}")
    rows = []
    for line in r.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != len(FIELDS):
            continue
        d = dict(zip(FIELDS, parts))
        d["cls"] = classify(d["JobName"])
        d["ng"] = gpus(d["AllocTRES"])
        d["cpu_s"] = to_int(d["CPUTimeRAW"])
        d["gpu_s"] = to_int(d["ElapsedRaw"]) * d["ng"]
        rows.append(d)
    if not rows:
        sys.exit("sacct returned no rows -- refusing to write an empty table")

    order = [MAIN, CHECK, LAUNCH, UNK]
    tot = {c: [0, 0, 0] for c in order}            # n, cpu_s, gpu_s
    state = {c: collections.Counter() for c in order}
    pref = {c: collections.defaultdict(lambda: [0, 0, 0]) for c in order}
    unk_names = collections.Counter()
    for d in rows:
        c = d["cls"]
        for acc in (tot[c], pref[c][prefix(d["JobName"])]):
            acc[0] += 1; acc[1] += d["cpu_s"]; acc[2] += d["gpu_s"]
        st = d["State"].split()[0] if d["State"] else "?"
        state[c][st] += 1
        if c == UNK:
            unk_names[d["JobName"]] += 1

    h = lambda s: f"{s / 3600:,.1f}"
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# Job accounting — {USER}, {START} → {now}", "",
         f"sacct -X --array (allocations + array elements; running jobs counted up to now; "
         f"pending jobs count 0 h). Rows: {len(rows)}. core-h = ΣCPUTimeRAW/3600; "
         f"GPU-h = Σ ElapsedRaw × gres/gpu / 3600.", "",
         "Classification is by job name (patterns in `job_accounting.py`); `mltp_<stage>` jobs "
         "other than `mltp_launch` count as 본 계산.", "",
         "## By class", "",
         "| class | jobs | core-h | GPU-h | states |", "|---|---:|---:|---:|---|"]
    gt = [0, 0, 0]
    for c in order:
        n, cs, gs = tot[c]
        gt = [gt[0] + n, gt[1] + cs, gt[2] + gs]
        sts = ", ".join(f"{k} {v}" for k, v in state[c].most_common())
        L.append(f"| {c} | {n} | {h(cs)} | {h(gs)} | {sts} |")
    L.append(f"| **total** | **{gt[0]}** | **{h(gt[1])}** | **{h(gt[2])}** | |")
    for c in order:
        if not tot[c][0]:
            continue
        L += ["", f"## {c} — top 15 name prefixes (by core-h + GPU-h)", "",
              "| prefix | jobs | core-h | GPU-h |", "|---|---:|---:|---:|"]
        items = sorted(pref[c].items(), key=lambda kv: -(kv[1][1] + kv[1][2]))
        for p, (n, cs, gs) in items[:15]:
            L.append(f"| `{p}` | {n} | {h(cs)} | {h(gs)} |")
        if len(items) > 15:
            rest = [sum(v[i] for _, v in items[15:]) for i in range(3)]
            L.append(f"| (other {len(items) - 15} prefixes) | {rest[0]} | {h(rest[1])} | {h(rest[2])} |")
    L += ["", "## Unclassified job names (for the lead to reclassify)", ""]
    if unk_names:
        L.append(", ".join(f"`{k}` ×{v}" for k, v in sorted(unk_names.items())))
    else:
        L.append("(none)")
    L.append("")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(L))
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(rows)} rows, unclassified names {len(unk_names)}")


if __name__ == "__main__":
    main()
