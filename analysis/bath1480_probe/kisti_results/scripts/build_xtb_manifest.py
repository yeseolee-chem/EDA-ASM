#!/usr/bin/env python3
"""
build_xtb_manifest.py — per-reaction XYZ prep for xtb_channel_generator.

For each rxn_XXXX/eda.inp:
  1. Parse ORCA input coordinates with fragment labels (C(1), N(2), etc.)
  2. Write 3 xyz files:
       ts.xyz        — all atoms (full TS complex)
       dp_dist.xyz   — fragment 1 atoms only (our convention: frag1 = dp)
       di_dist.xyz   — fragment 2 atoms only (our convention: frag2 = di, has N/O)
  3. Run xtb --opt --gfn 2 --alpb water on dp_dist.xyz -> dp_gs.xyz
  4. Run xtb --opt --gfn 2 --alpb water on di_dist.xyz -> di_gs.xyz
  5. Append row to manifest.csv

Idempotent: if all 5 xyz files exist for a reaction, skip.

Batch invocation via SLURM array — set --shard k --nshards N to process
subset of reactions (round-robin).

Usage:
    python build_xtb_manifest.py --root <eda_data_root> --out <xtb_workspace>
        [--shard 0 --nshards 10]
"""
from __future__ import annotations
import argparse, csv, subprocess, sys, tempfile
from pathlib import Path
import re

# eda.inp coord line: "  C(1)   1.234  5.678  9.012"
_COORD = re.compile(r"^\s*([A-Z][a-z]?)\((\d+)\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def parse_eda_inp(path: Path):
    """Return list of (elem, frag_id, x, y, z) tuples."""
    atoms = []
    in_coord = False
    for ln in path.read_text().splitlines():
        s = ln.strip()
        if s.startswith("* xyz"):
            in_coord = True; continue
        if in_coord and s == "*":
            break
        if in_coord:
            m = _COORD.match(ln)
            if m:
                atoms.append((m.group(1), int(m.group(2)),
                              float(m.group(3)), float(m.group(4)), float(m.group(5))))
    return atoms


def write_xyz(path: Path, atoms, title=""):
    lines = [str(len(atoms)), title]
    for e, _f, x, y, z in atoms:
        lines.append(f"{e} {x:.8f} {y:.8f} {z:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def xtb_opt(input_xyz: Path, output_xyz: Path, xtb_bin="xtb", solvent="water", timeout=600):
    """Run xtb --opt and copy final geometry to output_xyz."""
    if output_xyz.exists():
        return True
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # xtb writes xtbopt.xyz in cwd
        cmd = [xtb_bin, str(input_xyz.resolve()), "--gfn", "2", "--opt", "tight",
               "--chrg", "0", "--alpb", solvent]
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout)
        opt_file = td / "xtbopt.xyz"
        if not opt_file.exists():
            return False
        output_xyz.parent.mkdir(parents=True, exist_ok=True)
        output_xyz.write_bytes(opt_file.read_bytes())
    return True


def process_reaction(rxn_dir: Path, out_root: Path, xtb_bin="xtb", solvent="water"):
    """Prep 5 xyz files for one reaction. Returns dict with paths + status."""
    eda_inp = rxn_dir / "eda.inp"
    if not eda_inp.exists():
        return {"rxn_id": rxn_dir.name, "ok": False, "err": "no eda.inp"}

    rxn_id = int(re.search(r"\d+", rxn_dir.name).group())
    out_dir = out_root / rxn_dir.name

    ts_xyz = out_dir / "ts.xyz"
    dp_dist_xyz = out_dir / "dp_dist.xyz"
    di_dist_xyz = out_dir / "di_dist.xyz"
    dp_gs_xyz = out_dir / "dp_gs.xyz"
    di_gs_xyz = out_dir / "di_gs.xyz"

    # Idempotent: skip if all 5 exist
    if all(p.exists() for p in (ts_xyz, dp_dist_xyz, di_dist_xyz, dp_gs_xyz, di_gs_xyz)):
        return {"rxn_id": rxn_id, "ok": True, "cached": True,
                "ts_xyz": str(ts_xyz), "dp_dist_xyz": str(dp_dist_xyz),
                "di_dist_xyz": str(di_dist_xyz), "dp_gs_xyz": str(dp_gs_xyz),
                "di_gs_xyz": str(di_gs_xyz)}

    try:
        atoms = parse_eda_inp(eda_inp)
        frag1 = [a for a in atoms if a[1] == 1]  # dp (dipolarophile)
        frag2 = [a for a in atoms if a[1] == 2]  # di (dipole)
        assert len(frag1) > 0 and len(frag2) > 0

        write_xyz(ts_xyz, atoms, f"ts rxn_{rxn_id:04d}")
        write_xyz(dp_dist_xyz, frag1, f"dp_dist rxn_{rxn_id:04d} ({len(frag1)} atoms)")
        write_xyz(di_dist_xyz, frag2, f"di_dist rxn_{rxn_id:04d} ({len(frag2)} atoms)")

        if not xtb_opt(dp_dist_xyz, dp_gs_xyz, xtb_bin, solvent):
            return {"rxn_id": rxn_id, "ok": False, "err": "xtb opt failed for dp"}
        if not xtb_opt(di_dist_xyz, di_gs_xyz, xtb_bin, solvent):
            return {"rxn_id": rxn_id, "ok": False, "err": "xtb opt failed for di"}

        return {"rxn_id": rxn_id, "ok": True, "cached": False,
                "ts_xyz": str(ts_xyz), "dp_dist_xyz": str(dp_dist_xyz),
                "di_dist_xyz": str(di_dist_xyz), "dp_gs_xyz": str(dp_gs_xyz),
                "di_gs_xyz": str(di_gs_xyz)}
    except Exception as e:
        return {"rxn_id": rxn_id, "ok": False, "err": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir with rxn_XXXX/eda.inp")
    ap.add_argument("--out", required=True, help="xtb workspace dir")
    ap.add_argument("--manifest", default=None, help="path to manifest.csv (default: <out>/manifest.csv)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--xtb", default="xtb")
    ap.add_argument("--solvent", default="water")
    a = ap.parse_args()

    root = Path(a.root)
    out_root = Path(a.out)
    manifest_path = Path(a.manifest) if a.manifest else out_root / "manifest.csv"

    rxn_dirs = sorted(root.glob("rxn_*"))
    my_dirs = [d for i, d in enumerate(rxn_dirs) if i % a.nshards == a.shard]
    print(f"[shard {a.shard}/{a.nshards}] processing {len(my_dirs)} of {len(rxn_dirs)} reactions")

    rows = []
    n_ok, n_cached, n_fail = 0, 0, 0
    for i, rd in enumerate(my_dirs):
        r = process_reaction(rd, out_root, a.xtb, a.solvent)
        if r.get("ok"):
            n_ok += 1
            if r.get("cached"): n_cached += 1
            rows.append({k: r[k] for k in
                         ("rxn_id","ts_xyz","dp_dist_xyz","di_dist_xyz","dp_gs_xyz","di_gs_xyz")})
        else:
            n_fail += 1
            print(f"[FAIL] rxn_{r['rxn_id']:04d}: {r['err']}", file=sys.stderr, flush=True)
        if (i + 1) % 50 == 0:
            print(f"  progress: {i+1}/{len(my_dirs)}  ok={n_ok} cached={n_cached} fail={n_fail}", flush=True)

    # Write shard manifest (later merged)
    shard_manifest = out_root / f"manifest_shard{a.shard:02d}.csv"
    shard_manifest.parent.mkdir(parents=True, exist_ok=True)
    with shard_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["reaction_number","ts_xyz","dp_dist_xyz",
                                          "di_dist_xyz","dp_gs_xyz","di_gs_xyz",
                                          "charge_di","charge_dp"])
        w.writeheader()
        for r in rows:
            w.writerow({"reaction_number": r["rxn_id"], **{k: r[k] for k in
                        ("ts_xyz","dp_dist_xyz","di_dist_xyz","dp_gs_xyz","di_gs_xyz")},
                        "charge_di": 0, "charge_dp": 0})
    print(f"[shard {a.shard}] done. ok={n_ok} (cached={n_cached}) fail={n_fail}")
    print(f"[shard {a.shard}] manifest: {shard_manifest}")


if __name__ == "__main__":
    main()
