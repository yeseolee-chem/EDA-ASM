#!/usr/bin/env python3
"""make_reference.py — stage 'ref': strain-reference geometries (frag*_rel) for one job.

Port of Coley's dipole stereo-compatibility post-processing (github.com/coleygroup/
dipolar_cycloaddition_dataset, postprocess_reaction_profiles/postprocess_reaction_profiles.py
and finalize_reaction_profiles.py; Stuyver, Jorner & Coley, Sci. Data 10:66 (2023),
doi:10.1038/s41597-023-01977-8, Fig. 7). The D0 labels use Coley's r*_alt.xyz whenever it
exists, so D1 must apply the same rule to be on the same footing.

Coley's rule, step by step (our code follows it; deviations are marked D-n, see SPEC §5.3)
  1. dipole cut out of the TS; it and the original reactant dipole are GFN2-xTB optimised
     (water). Abort if the reactant's ring count differs from the SMILES; if the TS-derived
     dipole changed its ring count, keep its unoptimised geometry.
  2. dihedrals about the two dipole backbone bonds (none for triple/aromatic backbones).
     D-1: poles from the new ring, not from formal charges (Coley's charge loop picks a
     nitro N+/O- when the dipole carries NO2). D-2: a dihedral through an sp centre
     (angle > 165 deg) is not tracked.
  3. each dihedral scanned with xtb (--alpb water, force constant 0.1, 60 steps over 360
     deg); ROTATABLE iff |E_start - E_end| <= 5, max bias <= 20, and 2 <= barrier <= 20
     kcal/mol.
  4. 1000 randomise-and-relax conformers of the TS-derived dipole (autodE simanl), with the
     unrotatable dihedrals frozen by 4 distance constraints each; prune, xtb optimise, drop
     graph changes; keep the lowest if it beats the starting geometry.
  5. to_run: always if something was frozen ("constrained"); otherwise only if the new
     conformer is > 0.1 kcal/mol below the original at xtb.
  6. DFT (opt + freq + sp, same level as autodE) of the new conformer.
  7. use it ("alt") if constrained and heavy-atom RMSD to the original > 0.05 A;
     if unconstrained, if G(alt) < G(original) - 0.239 kcal/mol (G = E_sp + G_cont);
     a ring-count change in alt aborts the job (Coley drops the profile).
  D-3 (diagnostic only, never used for labels): step 6 is run even when to_run is False,
  so the pilot can report how often Coley's xtb gate hides a lower, TS-compatible reference.

D0_replay jobs take Coley's own choice (labels_all rel1_file/rel2_file) as the reference and
run steps 1-7 as a diagnostic in ref/port/ (agreement of the port with Coley's decision).
The dipolarophile reference is autodE's reactant unchanged (Coley enforces its stereo via
the product SMILES); its reacting C=C configuration is checked against the TS.
Writes: ref_dipole.xyz, ref_dipolarophile.xyz, ref_result.json, .done_ref | .fail_ref
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_common as pc  # noqa: E402
import stereo_utils as su  # noqa: E402

HARTREE_KCAL = pc.EH_TO_KCAL
N_RR = 1000


class StageFail(Exception):
    pass


# ---------------------------------------------------------------- helpers
def reorder_ts_fragment(fr, ts, P, r):
    """TS coordinates of fragment A, in the atom order of reactant r (heavy via map0, H via owner)."""
    ts_syms, ts_xyz = ts
    r_syms, r_xyz = r
    _, own_ts, _, _, _ = fr.heavy_graph(ts_syms, ts_xyz)
    _, own_r, _, _, _ = fr.heavy_graph(r_syms, r_xyz)
    pool = {}
    for h, o in own_ts.items():
        if h in P.A:
            pool.setdefault(o, []).append(h)
    out = np.zeros_like(r_xyz)
    for i, s in enumerate(r_syms):
        if s != "H":
            out[i] = ts_xyz[P.map0[i]]
    for i, s in enumerate(r_syms):
        if s == "H":
            cand = pool.get(P.map0[own_r[i]], [])
            if not cand:
                raise StageFail("H assignment failed while reordering the TS dipole")
            out[i] = ts_xyz[cand.pop(0)]
    return r_syms, out


def ade_mol(ade, name, syms, xyz, charge):
    atoms = [ade.Atom(s, *c) for s, c in zip(syms, xyz)]
    return ade.Molecule(name=name, atoms=atoms, charge=int(charge), mult=1, solvent_name="water")


def n_cycles(mol):
    from autode.mol_graphs import find_cycles
    return len(find_cycles(mol.graph))


def dihedral_of(xyz, q):
    return su.dihedral_deg(*(xyz[i] for i in q))


def xtb_scan(xtb_bin, workdir: Path, syms, xyz, quad, charge):
    """Coley write_scan_input_file + assess_bond_rotatability_from_scan (xtb, ALPB water)."""
    workdir.mkdir(parents=True, exist_ok=True)
    pc.write_xyz(workdir / "in.xyz", syms, xyz)
    phi0 = round(dihedral_of(xyz, quad), 2)
    a, b, c, d = (i + 1 for i in quad)
    (workdir / "scan.inp").write_text(
        f"$constrain\n force constant=0.1\n dihedral: {a},{b},{c},{d},{phi0}\n$end\n"
        f"$scan\n1:{phi0},{phi0 + 360},60\n$end")
    out = workdir / "scan.out"
    with open(out, "w") as fo:
        subprocess.run([xtb_bin, "in.xyz", "--alpb", "water", "--opt", "--chrg", str(int(charge)),
                        "--input", "scan.inp"], cwd=workdir, stdout=fo, stderr=subprocess.STDOUT, check=False)
    lines = out.read_text(errors="replace").splitlines()
    E = [float(l.split()[2]) for l in lines if l.startswith("unbiased energy:")]
    B = [float(l.split()[2]) for l in lines if l.startswith("    bias energy:")]
    if len(E) < 2 or not B:
        return dict(quad=quad, phi0=phi0, n_points=len(E), rotatable=None, error="scan output not parsed")
    k = HARTREE_KCAL
    stats = dict(end_diff=abs(E[0] - E[-1]) * k, max_bias=max(B) * k, barrier=(max(E) - min(E)) * k)
    rot = not (stats["end_diff"] > 5 or stats["max_bias"] > 20 or stats["barrier"] > 20 or stats["barrier"] < 2)
    return dict(quad=quad, phi0=phi0, n_points=len(E), rotatable=rot, **stats)


def dft_refine(ade, mol, temp):
    from autode.methods import get_hmethod
    h = get_hmethod()
    mol.optimise(method=h)
    mol.calc_thermo(method=h, temp=temp)
    mol.single_point(method=h)
    e_sp = float(mol.energies.last_potential)
    g = float(mol.g_cont) if mol.g_cont is not None else None
    return dict(e_opt_eh=float(mol.energies.first_potential), e_sp_eh=e_sp, g_cont_eh=g,
                G_eh=(e_sp + g) if g is not None else None)


def heavy_rmsd(ade, a_syms, a_xyz, b_xyz):
    from autode.geom import calc_rmsd
    idx = [i for i, s in enumerate(a_syms) if s != "H"]
    return float(calc_rmsd(np.asarray(a_xyz)[idx], np.asarray(b_xyz)[idx]))


# ---------------------------------------------------------------- Coley port
def coley_port(cfg, jd, work: Path, ts_res, fr, orig, orig_energy, dry):
    """Steps 1-7. `orig` = (syms, xyz) of the reactant dipole the profile used; `orig_energy`
    = dict(e_sp_eh, g_cont_eh). Returns the decision record and the alt geometry."""
    ade = pc.configure_autode(cfg)
    if dry:
        from run_ts import dry_xtb_patch
        dry_xtb_patch()
    from autode.conformers.conf_gen import get_simanl_conformer
    temp = cfg["ts_stage"]["temp_K"]
    q = int(ts_res["charge_dipole"])
    dip_smi, bb = ts_res["dipole_mapped_smiles"], tuple(ts_res["dipole_backbone_maps"])
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    rec = dict(port="coley_2023", deviations=["D-1", "D-2", "D-3"])
    if orig_energy is None:
        orig_energy = dft_refine(ade, ade_mol(ade, "orig_dft", *orig, q), temp)
        rec["orig_recomputed"] = orig_energy

    ts = pc.read_xyz(jd / "ts.xyz")
    r_dph = pc.read_xyz(jd / "r_dipolarophile_autode.xyz")
    P = fr.partition(ts, orig, r_dph)
    if P.status != "ok":
        raise StageFail(f"partition with the reference dipole: {P.status}")
    ts_syms, ts_dip = reorder_ts_fragment(fr, ts, P, orig)
    pc.write_xyz(work / "ts_dipole_extracted.xyz", ts_syms, ts_dip)

    # 1. xtb optimisations
    xtb = ade.methods.XTB()
    r_mol = ade_mol(ade, "orig_xtb", *orig, q); r_mol.optimise(method=xtb); r_mol.reset_graph()
    plain = su.Chem.MolToSmiles(_unmapped(dip_smi))
    s_mol = ade.Molecule(name="smiles_ref", smiles=plain)
    if n_cycles(r_mol) != n_cycles(s_mol):
        rec.update(abort="reactant ring count differs from SMILES", use_alt=False)
        return rec, None
    t_mol = ade_mol(ade, "tsdip_xtb", ts_syms, ts_dip, q); t_mol.optimise(method=xtb); t_mol.reset_graph()
    if n_cycles(t_mol) != n_cycles(r_mol):
        t_mol = ade_mol(ade, "tsdip_raw", ts_syms, ts_dip, q)              # Coley: fall back to the raw cut
        t_mol.single_point(method=xtb)
        rec["ts_dipole_opt_changed_rings"] = True
    t_xyz = np.array([a.coord for a in t_mol.atoms], dtype=float)
    e_start_xtb = float(t_mol.energy)

    # 2. dihedrals to track (geometry indices in reactant order)
    blk = su.xyz_block_from_atoms(ts_syms, t_xyz)
    tag, dih, quads = su.config_tags(blk, dip_smi, bb, q)
    quads = [qd for qd, tg in zip(quads, tag) if tg != "?"]
    rec.update(tracked=quads, ts_dipole_config=tag)

    # 3. rotatability scans
    scans = [xtb_scan(ade.Config.XTB.path, work / f"scan_{k}", ts_syms, t_xyz, qd, q) for k, qd in enumerate(quads)]
    rec["scans"] = scans
    if any(s["rotatable"] is None for s in scans):
        raise StageFail("xtb rotatability scan could not be parsed (check xtb version / output format)")
    frozen = [s["quad"] for s in scans if not s["rotatable"]]

    # 4. randomise-and-relax conformers with the unrotatable dihedrals frozen
    dist = {}
    for a, b, c, d in frozen:
        for i, j in ((a, b), (b, c), (c, d), (a, d)):
            dist[(i, j)] = float(np.linalg.norm(t_xyz[i] - t_xyz[j]))
    rr = work / "rr"; rr.mkdir(exist_ok=True); os.chdir(rr)
    t0 = time.time()
    for n in range(N_RR):
        t_mol.conformers.append(get_simanl_conformer(t_mol, dist_consts=dist or None, conf_n=n))
    t_mol.conformers.prune_on_energy(e_tol=1e-10)
    t_mol.conformers.prune_on_rmsd()
    t_mol.conformers.optimise(method=xtb)
    t_mol.conformers.prune(remove_no_energy=True)
    t_mol.conformers.prune_diff_graph(t_mol.graph)
    n_conf = len(t_mol.conformers)
    if n_conf and t_mol.energy > t_mol.conformers.lowest_energy.energy:
        t_mol._set_lowest_energy_conformer()
    os.chdir(work); shutil.rmtree(rr, ignore_errors=True)
    dE_xtb = (float(t_mol.energy) - float(r_mol.energy)) * HARTREE_KCAL
    rec.update(n_rr_conformers_kept=n_conf, rr_wall_s=time.time() - t0, dE_xtb_alt_minus_orig=dE_xtb,
               e_start_xtb=e_start_xtb)

    # 5. to_run / constrained
    constrained = bool(frozen)
    to_run = True if constrained else dE_xtb < -0.1
    rec.update(constrained=constrained, to_run=to_run)

    # 6. DFT of alt (always: D-3 diagnostic when to_run is False)
    alt = ade_mol(ade, "dipole_alt", [a.label for a in t_mol.atoms],
                  np.array([a.coord for a in t_mol.atoms], dtype=float), q)
    rec["alt_dft"] = dft_refine(ade, alt, temp)
    alt.reset_graph()
    a_syms, a_xyz = [a.label for a in alt.atoms], np.array([a.coord for a in alt.atoms], dtype=float)
    pc.write_xyz(work / "dipole_alt.xyz", a_syms, a_xyz, "Coley-port alt dipole (DFT)")
    o_mol = ade_mol(ade, "orig_graph", *orig, q)
    cyclization = n_cycles(alt) != n_cycles(o_mol)
    same_graph = alt.graph.is_isomorphic_to(o_mol.graph) if hasattr(alt.graph, "is_isomorphic_to") else None
    rmsd = heavy_rmsd(ade, a_syms, a_xyz, orig[1])
    G_orig = (orig_energy["e_sp_eh"] + orig_energy["g_cont_eh"]) if orig_energy.get("g_cont_eh") is not None else None
    dG = (rec["alt_dft"]["G_eh"] - G_orig) * HARTREE_KCAL if (G_orig is not None and rec["alt_dft"]["G_eh"]) else None
    dE_sp = (rec["alt_dft"]["e_sp_eh"] - orig_energy["e_sp_eh"]) * HARTREE_KCAL
    alt_cfg = su.config_tags(su.xyz_block_from_atoms(a_syms, a_xyz), dip_smi, bb, q)[0]
    rec.update(cyclization=cyclization, alt_same_graph=same_graph, rmsd_heavy_alt_orig=rmsd,
               dG_alt_minus_orig_kcal=dG, dEsp_alt_minus_orig_kcal=dE_sp, alt_config=alt_cfg)

    # 7. Coley's decision
    if cyclization:
        rec.update(use_alt=False, abort="ring count changed in alt (Coley drops such profiles)")
        return rec, (a_syms, a_xyz)
    if same_graph is False:
        rec.update(use_alt=False, abort="alt connectivity differs from the reactant")
        return rec, (a_syms, a_xyz)
    if to_run and constrained:
        use = rmsd > 0.05
        why = f"constrained; RMSD {rmsd:.3f} {'>' if use else '<='} 0.05 A"
    elif to_run:
        use = dG is not None and dG < -0.239
        why = f"unconstrained; dG {dG if dG is None else round(dG, 3)} vs -0.239 kcal/mol"
    else:
        use = False
        why = f"xtb gate: dE_xtb {dE_xtb:.3f} >= -0.1 kcal/mol"
    # D-3 diagnostic: what a "lowest TS-compatible" rule would have picked
    orig_cfg = su.config_tags(su.xyz_block_from_atoms(*orig), dip_smi, bb, q)[0]
    strict = (not su.tags_compatible(orig_cfg, tag)) or (dG is not None and dG < -0.239)
    rec.update(use_alt=bool(use), reason=why, orig_config=orig_cfg, strict_rule_would_use_alt=bool(strict))
    return rec, (a_syms, a_xyz)


def _unmapped(smi):
    m = su.Chem.MolFromSmiles(smi)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return m


def d0_energies(jd: Path, stem: str):
    """E_sp / G_cont of a species from Coley's energies.csv (Species,E_opt,G_cont,H_cont,E_sp)."""
    for line in (jd / "energies.csv").read_text().splitlines()[2:]:
        t = line.split(",")
        if t[0] == stem:
            return dict(e_opt_eh=float(t[1]), g_cont_eh=float(t[2]), h_cont_eh=float(t[3]), e_sp_eh=float(t[4]))
    raise StageFail(f"{stem} not in energies.csv")


def dipolarophile_check(fr, jd, ts_res, job, ref_dph):
    """Configuration of the dipolarophile's reacting bond: TS vs reference (must agree)."""
    ts = pc.read_xyz(jd / "ts.xyz")
    B = ts_res["B_idx"]
    rmols, formed, _ = fr.smiles_reactants_and_formed(job["mapped_smiles"])
    dip = fr.dipole_smiles_index(job["mapped_smiles"])[0]
    dmol = rmols[1 - dip]
    maps = {a.GetAtomMapNum() for a in dmol.GetAtoms()}
    react = sorted({m for pair in formed for m in pair if m in maps})
    if len(react) != 2:
        return dict(checked=False, why="reacting atoms not found")
    smi = su.Chem.MolToSmiles(dmol)
    tmpl = su._clean_template(smi)
    idx = {a.GetAtomMapNum(): a.GetIdx() for a in tmpl.GetAtoms() if a.GetAtomMapNum()}
    u, v = idx[react[0]], idx[react[1]]
    bond = tmpl.GetBondBetweenAtoms(u, v)
    if bond is None or bond.GetBondType() != su.Chem.BondType.DOUBLE:
        return dict(checked=False, why="reacting bond is not a double bond")
    plain = su.Chem.Mol(tmpl)
    for at in plain.GetAtoms():
        at.SetAtomMapNum(0)                               # map numbers would break the symmetry ranking
    ranks = list(su.Chem.CanonicalRankAtoms(plain, breakTies=False))

    def ref_nb(x, y):
        nb = [n.GetIdx() for n in tmpl.GetAtomWithIdx(x).GetNeighbors() if n.GetIdx() != y]
        if len(nb) < 2 or ranks[nb[0]] == ranks[nb[1]]:
            return None                                   # two identical substituents: not stereogenic
        return max(nb, key=lambda i: ranks[i])
    nu, nv = ref_nb(u, v), ref_nb(v, u)
    if nu is None or nv is None:
        return dict(checked=False, why="reacting C=C not stereogenic")

    def tag(syms, xyz, charge):
        m, t2g, _ = su.template_match(su.xyz_block_from_atoms(syms, xyz), smi, charge)
        p = m.GetConformer().GetPositions()
        return "c" if abs(su.dihedral_deg(*(p[t2g[i]] for i in (nu, u, v, nv)))) < 90 else "t"
    q = int(ts_res["charge_dipolarophile"])
    t_ts = tag([ts[0][i] for i in B], ts[1][B], q)
    t_ref = tag(*ref_dph, q)
    return dict(checked=True, ts=t_ts, ref=t_ref, ok=t_ts == t_ref)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--config"); ap.add_argument("--manifest")
    ap.add_argument("--dry-xtb", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--skip-port", action="store_true", help="replay jobs: do not run the port diagnostic")
    a = ap.parse_args()
    cfg = pc.load_config(a.config); job = pc.load_job(a.job, a.manifest); jd = pc.job_dir(cfg, a.job)
    if pc.done(jd, "ref"):
        print(f"{a.job}: ref already done"); return
    if not pc.done(jd, "ts"):
        raise SystemExit(f"{a.job}: ts stage not done")
    fr, _ = pc.import_repo_modules(cfg)
    ts_res = json.loads((jd / "ts_result.json").read_text())
    res = dict(job_id=a.job, kind=job["kind"])
    t0 = time.time()
    try:
        orig = pc.read_xyz(jd / "r_dipole_autode.xyz")
        r_dph = pc.read_xyz(jd / "r_dipolarophile_autode.xyz")
        if job["kind"] == "D0_replay":
            rel1, rel2 = job["d0_rel1_file"], job["d0_rel2_file"]     # frag1 = dipole in labels_all
            ref_dip = pc.read_xyz(jd / f"d0_{rel1}")
            r_dph = pc.read_xyz(jd / f"d0_{rel2}")
            res.update(source="d0_labels_all", rel1_file=rel1, rel2_file=rel2, alt_used=int("_alt" in rel1))
            if not a.skip_port:
                try:
                    port, _ = coley_port(cfg, jd, jd / "ref" / "port", ts_res, fr, orig, None, a.dry_xtb)
                except StageFail as e:
                    port = dict(error=str(e))
                except Exception as e:                            # noqa: BLE001
                    port = dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
                port["agrees_with_coley"] = (port.get("use_alt") == bool(res["alt_used"])) if "use_alt" in port else None
                res["port_diagnostic"] = port
        else:
            k = ts_res["reactant_index_dipole"]
            e_orig = ts_res["reactants"][k]
            port, alt = coley_port(cfg, jd, jd / "ref", ts_res, fr, orig, e_orig, a.dry_xtb)
            res["port"] = port
            if "abort" in port and port.get("abort", "").startswith("ring count changed"):
                raise StageFail(port["abort"])
            ref_dip = alt if port["use_alt"] else orig
            res.update(source="coley_port", alt_used=int(port["use_alt"]))
        res["dipolarophile_check"] = dipolarophile_check(fr, jd, ts_res, job, r_dph)
        if res["dipolarophile_check"].get("ok") is False:
            raise StageFail("dipolarophile configuration at the TS differs from its reference")
    except StageFail as e:
        res["wall_s"] = time.time() - t0
        pc.write_json(jd / "ref_result.json", res)
        pc.mark_fail(jd, "ref", str(e)); return
    except Exception as e:                                        # noqa: BLE001
        res.update(wall_s=time.time() - t0, traceback=traceback.format_exc())
        pc.write_json(jd / "ref_result.json", res)
        pc.mark_fail(jd, "ref", f"{type(e).__name__}: {e}"); return
    pc.write_xyz(jd / "ref_dipole.xyz", *ref_dip, f"{a.job} dipole reference ({res['source']}, alt_used={res['alt_used']})")
    pc.write_xyz(jd / "ref_dipolarophile.xyz", *r_dph, f"{a.job} dipolarophile reference")
    res["wall_s"] = time.time() - t0
    pc.write_json(jd / "ref_result.json", res)
    pc.mark_done(jd, "ref")
    p = res.get("port") or res.get("port_diagnostic") or {}
    print(f"{a.job}: ref ok  source={res['source']} alt_used={res['alt_used']}  "
          f"port: frozen={len([s for s in p.get('scans', []) if s.get('rotatable') is False])} "
          f"to_run={p.get('to_run')} use_alt={p.get('use_alt')} ({p.get('reason', p.get('error', ''))})")


if __name__ == "__main__":
    main()
