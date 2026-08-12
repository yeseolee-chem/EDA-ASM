#!/usr/bin/env python3
"""Per-reaction status snapshot for the DFT pilot.

Also writes work/rxn_XXXX/summary.md into each reaction dir so the
workspace tree shows a readable per-reaction file.
"""
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/dft_pilot")
WORK = BASE / "work"

CYCLE_RE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)")
SCF_RE = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
CONV_RE = re.compile(r"OPTIMIZATION HAS CONVERGED")
IMAG_RE = re.compile(r"\*\*\*imaginary mode\*\*\*")
NORMAL_RE = re.compile(r"ORCA TERMINATED NORMALLY")
TOTAL_TIME_RE = re.compile(r"TOTAL RUN TIME:\s*([\d\s]+?)days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+)\s*seconds")

STAGES = ["dft_optts", "eda_tz"]


def read_head_tail(path: Path, n_head=0, n_tail=5000):
    """Return last n_tail chars of file (cheap for large ORCA outputs)."""
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - n_tail))
        return f.read().decode("utf-8", errors="replace")


def stage_status(rxn_dir: Path, stage: str):
    inp = rxn_dir / f"{stage}.inp"
    out = rxn_dir / f"{stage}.out"
    done = rxn_dir / f"{stage}.done"

    st = {"inp_exists": inp.exists(), "out_exists": out.exists(),
          "done_sentinel": done.exists(), "size_bytes": 0,
          "opt_cycles": 0, "scf_converged": 0, "opt_done": False,
          "orca_normal": False, "n_imag": None, "total_seconds": None}

    if not out.exists():
        return st
    st["size_bytes"] = out.stat().st_size
    text = out.read_text(errors="replace")
    st["opt_cycles"] = len(CYCLE_RE.findall(text))
    st["scf_converged"] = len(SCF_RE.findall(text))
    st["opt_done"] = bool(CONV_RE.search(text))
    st["orca_normal"] = bool(NORMAL_RE.search(text))
    imags = IMAG_RE.findall(text)
    st["n_imag"] = len(imags)
    m = TOTAL_TIME_RE.search(text)
    if m:
        d = int(m.group(1).strip() or 0)
        h, mi, sec = int(m.group(2)), int(m.group(3)), int(m.group(4))
        st["total_seconds"] = d * 86400 + h * 3600 + mi * 60 + sec
    return st


def format_time(seconds):
    if seconds is None: return "-"
    if seconds < 60:  return f"{seconds}s"
    if seconds < 3600: return f"{seconds//60}m{seconds%60}s"
    return f"{seconds//3600}h{(seconds%3600)//60}m"


def rxn_row(rxn_dir: Path):
    rxn_id = int(rxn_dir.name.replace("rxn_", ""))
    labels = json.loads((rxn_dir / "labels.json").read_text())
    n_atoms = len(labels["elements"])
    opt = stage_status(rxn_dir, "dft_optts")
    eda = stage_status(rxn_dir, "eda_tz")

    channels_path = rxn_dir / "channels_dft.json"
    channels = None
    if channels_path.exists():
        channels = json.loads(channels_path.read_text())

    return {
        "rxn_id": rxn_id, "n_atoms": n_atoms,
        "opt": opt, "eda": eda, "channels": channels,
    }


def write_rxn_summary(rxn_dir: Path, row):
    lines = [f"# rxn_{row['rxn_id']:04d} — DFT pilot status",
             "", f"- atoms: **{row['n_atoms']}**",
             f"- generated: {datetime.now().isoformat(timespec='seconds')}",
             "", "## Stage 1: DFT TS reopt (B3LYP-D3BJ/def2-SVP OptTS+NumFreq)"]
    o = row["opt"]
    lines += [
        f"- OptTS opt cycles: **{o['opt_cycles']}**",
        f"- SCF converged banners: {o['scf_converged']}",
        f"- OPTIMIZATION HAS CONVERGED: {'✓' if o['opt_done'] else '—'}",
        f"- ORCA TERMINATED NORMALLY: {'✓' if o['orca_normal'] else '—'}",
        f"- imaginary modes seen: {o['n_imag'] if o['n_imag'] is not None else '-'}",
        f"- output size: {o['size_bytes']/1024:.1f} KB",
        f"- ORCA total wall: {format_time(o['total_seconds'])}",
        f"- sentinel `dft_optts.done`: {'✓' if o['done_sentinel'] else '—'}",
    ]

    lines += ["", "## Stage 2: EDA-NOCV SPE (B3LYP-D3BJ/def2-TZVP)"]
    e = row["eda"]
    if not e["inp_exists"]:
        lines.append("- (not started — waiting for OptTS to finish)")
    else:
        lines += [
            f"- ORCA TERMINATED NORMALLY: {'✓' if e['orca_normal'] else '—'}",
            f"- output size: {e['size_bytes']/1024:.1f} KB",
            f"- ORCA total wall: {format_time(e['total_seconds'])}",
            f"- sentinel `eda_tz.done`: {'✓' if e['done_sentinel'] else '—'}",
        ]

    lines += ["", "## Stage 3: parsed channels (kcal/mol)"]
    c = row["channels"]
    if c is None:
        lines.append("- (not parsed yet — waiting for EDA to finish)")
    else:
        lines += [
            f"- ok: **{c.get('ok')}**",
            f"- sum residual: {c.get('sum_residual_kcal','?'):.4f}",
            "",
            "| channel | value (kcal/mol) |",
            "|---|---:|",
        ]
        for ch in ["elst_dft", "pauli_dft", "oi_dft", "disp_dft",
                    "cpcm_dft", "cds_dft", "eint_dft"]:
            if ch in c:
                lines.append(f"| {ch} | {c[ch]:+.4f} |")

    (rxn_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    rxn_dirs = sorted(WORK.glob("rxn_*"))
    rows = [rxn_row(d) for d in rxn_dirs]

    # Print terminal snapshot
    print(f"{'='*100}")
    print(f"DFT pilot status — {datetime.now().isoformat(timespec='seconds')}")
    try:
        sq = subprocess.check_output(["squeue", "-u", "yeseo1ee", "-h",
                                       "-o", "%.10i %.2t %.6M %R"], text=True)
        print("SLURM tasks:")
        for line in sq.strip().splitlines():
            print(f"  {line}")
    except Exception as e:
        print(f"squeue error: {e}")

    print(f"{'='*100}")
    print(f"{'rxn':<8} {'atoms':>5} {'opt_cycles':>10} {'scf_conv':>8} "
          f"{'opt_done':>8} {'eda_done':>8} {'parsed':>7} "
          f"{'eint_dft':>10} {'opt_wall':>10}")
    print("-" * 100)
    for r in rows:
        eint = "-"
        if r["channels"] and "eint_dft" in r["channels"]:
            eint = f"{r['channels']['eint_dft']:+.2f}"
        print(f"rxn_{r['rxn_id']:04d} {r['n_atoms']:>5} "
              f"{r['opt']['opt_cycles']:>10} "
              f"{r['opt']['scf_converged']:>8} "
              f"{'✓' if r['opt']['opt_done'] else '—':>8} "
              f"{'✓' if r['eda']['done_sentinel'] else '—':>8} "
              f"{'✓' if r['channels'] else '—':>7} "
              f"{eint:>10} "
              f"{format_time(r['opt']['total_seconds']):>10}")

    # Write per-rxn summary.md files
    print(f"{'-'*100}")
    for d, r in zip(rxn_dirs, rows):
        write_rxn_summary(d, r)
        print(f"  wrote {d.name}/summary.md")


if __name__ == "__main__":
    main()
