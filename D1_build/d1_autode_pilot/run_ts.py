#!/usr/bin/env python3
"""run_ts.py — stage 'ts': autodE reaction profile for one manifest job (Coley 2023 protocol).

  python run_ts.py --job J03 [--config config.yaml] [--manifest pilot_manifest.csv]

Job kinds (manifest column `kind`):
  D1, D0_control  autodE computes the profile from the reaction SMILES
  D0_replay       no autodE: the D0 (Coley) TS and reactant geometries are copied in, so the
                  later stages can be checked against labels_all.json on identical geometry

Writes into <scratch>/jobs/<job>/ :
  ts/<job>/...                 autodE working tree (autodE checkpoints; a rerun resumes)
  ts.xyz                       lowest-energy TS conformer
  r_dipole_autode.xyz          autodE's dipole reactant (xtb-ranked conformer, DFT-optimised)
  r_dipolarophile_autode.xyz   idem, dipolarophile
  product.xyz, energies.csv    as printed by autodE
  ts_result.json               energies, frequencies, forming bonds, TS checks, keywords
  .done_ts | .fail_ts
Fails (-> .fail_ts) on: autodE exception, no TS, n_imag != 1, graph partition not ok,
forming-bond audit impossible, foreign inter-fragment bond, imaginary mode not on the
forming bonds. Flagged, not failed: asynchronous TS (a forming bond > 3.0 A); regiochemistry
ambiguous (crossed terminus pairing within 0.3 A of the SMILES pairing; autodE itself already
checks that the imaginary mode forms the product bonds).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot_common as pc  # noqa: E402
import stereo_utils as su  # noqa: E402


def species_record(mol) -> dict:
    e = mol.energies
    f = lambda v: float(v) if v is not None else None          # noqa: E731
    return dict(name=mol.name, n_atoms=mol.n_atoms, charge=mol.charge, mult=mol.mult,
                e_opt_eh=f(e.first_potential), e_sp_eh=f(e.last_potential),
                g_cont_eh=f(mol.g_cont), h_cont_eh=f(mol.h_cont))


def sym_xyz(mol):
    return [a.label for a in mol.atoms], np.array([a.coord for a in mol.atoms], dtype=float)


def forming_share(X, mode, formed_pairs, heavy):
    """Share of the imaginary-mode bond stretching carried by the forming bonds (top-2 normalised)."""
    def ddot(i, j):
        r = X[i] - X[j]
        return float(np.dot(r, mode[i] - mode[j]) / np.linalg.norm(r))
    allp = [(i, j) for k, i in enumerate(heavy) for j in heavy[k + 1:] if np.linalg.norm(X[i] - X[j]) < 3.5]
    vals = sorted((abs(ddot(i, j)) for i, j in allp), reverse=True)
    return sum(abs(ddot(i, j)) for i, j in formed_pairs) / (sum(vals[:2]) or 1e-12)


def dry_xtb_patch():
    """LOCAL TESTING ONLY: use GFN2-xTB as the 'high-level' method (no ORCA needed)."""
    import autode as ade
    import autode.methods as M
    ade.Config.XTB.keywords.sp = ["--sp"]          # print_output() asserts hmethod sp/opt keywords
    ade.Config.XTB.keywords.opt = ["--opt"]
    xtb = M.XTB
    M.get_hmethod = lambda: xtb()
    for name, mod in list(sys.modules.items()):
        if name.startswith("autode") and hasattr(mod, "get_hmethod"):
            setattr(mod, "get_hmethod", M.get_hmethod)


# ---------------------------------------------------------------- producers
def run_autode(cfg, job, jd, res, dry):
    """-> dict(ts=(syms,xyz), reacs=[(syms,xyz),..], prod=(syms,xyz)|None, imag=[..], mode=array|None)"""
    ade = pc.configure_autode(cfg)
    if dry:
        dry_xtb_patch()
    t = cfg["ts_stage"]
    res.update(autode_version=ade.__version__, keywords=pc.hmethod_keywords_summary(), n_cores=ade.Config.n_cores)
    wd = jd / "ts"; wd.mkdir(exist_ok=True)
    os.chdir(wd)
    try:
        rxn = ade.Reaction(job["rxn_smiles"], name=job["job_id"], solvent_name=t["solvent"], temp=t["temp_K"])
        rxn.calculate_reaction_profile(free_energy=bool(t["free_energy"]))
    except Exception as e:                                       # noqa: BLE001
        res["traceback"] = traceback.format_exc()
        raise StageFail(f"autodE exception: {type(e).__name__}: {e}")
    for f in ("energies.csv", "methods.txt"):
        p = wd / job["job_id"] / f
        if p.is_file():
            shutil.copy(p, jd / f)
    if rxn.ts is None:
        raise StageFail("autodE found no transition state")
    ts = rxn.ts
    res.update(ts=species_record(ts), reactants=[species_record(m) for m in rxn.reacs],
               products=[species_record(m) for m in rxn.prods])
    for k, key in (("dE_act_sp_kcal", "E"), ("dG_act_kcal", "G"), ("dH_act_kcal", "H")):
        try:
            v = rxn.delta(f"{key}‡")
            res[k] = float(v.to("kcal mol-1")) if v is not None else None
        except Exception:                                        # noqa: BLE001
            res[k] = None
    try:
        mode = np.array(ts.normal_mode(6), dtype=float)
    except Exception:                                            # noqa: BLE001
        mode = None
    if len(rxn.reacs) != 2:
        raise StageFail(f"{len(rxn.reacs)} reactants")
    return dict(ts=sym_xyz(ts), reacs=[sym_xyz(m) for m in rxn.reacs],
                prod=sym_xyz(rxn.prods[0]) if rxn.prods else None,
                imag=[float(v) for v in (ts.imaginary_frequencies or [])], mode=mode)


def load_replay(cfg, job, jd, res):
    """D0_replay: Coley's own TS / reactants (profile directory of figshare 21707888)."""
    pdir = Path(cfg["d0_profiles"]) / str(int(float(job["d0_rxn_id"])))
    ts_f = sorted(p for p in pdir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz")[0]
    rs = sorted(p for p in pdir.glob("r*.xyz") if "_alt" not in p.name)
    ps = sorted(pdir.glob("p*.xyz"))
    for p in pdir.glob("*.xyz"):
        shutil.copy(p, jd / ("d0_" + p.name))
    shutil.copy(pdir / "energies.csv", jd / "energies.csv")
    res.update(replay_profile=str(pdir), replay_ts_file=ts_f.name, replay_reactant_files=[p.name for p in rs])
    return dict(ts=pc.read_xyz(ts_f), reacs=[pc.read_xyz(p) for p in rs],
                prod=pc.read_xyz(ps[0]) if ps else None, imag=None, mode=None)


class StageFail(Exception):
    pass


# ---------------------------------------------------------------- common analysis
def analyse(cfg, job, jd, res, got, fr):
    g = cfg["gates"]
    ts_syms, ts_xyz = got["ts"]
    pc.write_xyz(jd / "ts.xyz", ts_syms, ts_xyz, f"{job['job_id']} TS")
    if got["prod"] is not None:
        pc.write_xyz(jd / "product.xyz", *got["prod"], f"{job['job_id']} product")
    imag = got["imag"]
    res.update(imag_freqs_cm=imag, n_imag=None if imag is None else len(imag))

    # which reactant is the dipole: graph match against the mapped SMILES
    r0, r1 = got["reacs"][:2]
    rmols, formed, _ = fr.smiles_reactants_and_formed(job["mapped_smiles"])
    assign = fr.match_smiles_to_reactants(rmols, r0, r1)
    dip = fr.dipole_smiles_index(job["mapped_smiles"])
    if assign is None or dip is None:
        raise StageFail("cannot match reactant geometries to the mapped SMILES")
    k_dip = 0 if assign[0] == dip[0] else 1
    r_dip, r_dph = (r0, r1) if k_dip == 0 else (r1, r0)
    pc.write_xyz(jd / "r_dipole_autode.xyz", *r_dip, f"{job['job_id']} dipole reactant")
    pc.write_xyz(jd / "r_dipolarophile_autode.xyz", *r_dph, f"{job['job_id']} dipolarophile reactant")
    q_dip, q_dph = fr.formal_charges(rmols, (dip[0], 1 - dip[0]))
    res.update(reactant_index_dipole=k_dip, charge_dipole=q_dip, charge_dipolarophile=q_dph)

    P = fr.partition((ts_syms, ts_xyz), r_dip, r_dph)
    res["partition"] = dict(status=P.status, **P.diag)
    if P.status != "ok":
        raise StageFail(f"partition:{P.status}")
    audit = fr.audit_forming_bonds(P, (ts_syms, ts_xyz), r_dip, r_dph, job["mapped_smiles"])
    if audit is None:
        raise StageFail("forming_bond_map_failed")
    heavy = [i for i, s in enumerate(ts_syms) if s != "H"]
    share = None if got["mode"] is None else forming_share(ts_xyz, got["mode"], sorted(audit.formed_ts), heavy)
    res.update(A_idx=sorted(P.A), B_idx=sorted(P.B), formed_pairs_ts=sorted(audit.formed_ts),
               formed_d=audit.d_formed, foreign=audit.foreign, imag_mode_forming_share=share)

    # dipole configuration at the TS (what the reference step has to match)
    try:
        _, bb, dip_smi = su.dipole_backbone(job["mapped_smiles"])
        A = sorted(P.A)
        tag, dih, _ = su.config_tags(su.xyz_block_from_atoms([ts_syms[i] for i in A], ts_xyz[A]), dip_smi, bb, q_dip)
        res.update(dipole_backbone_maps=bb, dipole_mapped_smiles=dip_smi, ts_dipole_config=tag,
                   ts_dipole_dihedrals=dih,
                   reactant_dipole_config=su.config_tags(su.xyz_block_from_atoms(*r_dip), dip_smi, bb, q_dip)[0])
    except Exception as e:                                       # noqa: BLE001
        res["config_error"] = f"{type(e).__name__}: {e}"

    # regiochemistry: the SMILES forming bonds must be shorter (summed) than the crossed pairing
    Ah = [i for i in sorted(P.A) if ts_syms[i] != "H"]; Bh = [i for i in sorted(P.B) if ts_syms[i] != "H"]
    contacts = sorted((float(np.linalg.norm(ts_xyz[i] - ts_xyz[j])), tuple(sorted((i, j)))) for i in Ah for j in Bh)
    fset = {tuple(sorted(p)) for p in audit.formed_ts}
    (i, j), (k, l) = [(p if p[0] in P.A else (p[1], p[0])) for p in sorted(audit.formed_ts)]
    dist = lambda u, v: float(np.linalg.norm(ts_xyz[u] - ts_xyz[v]))     # noqa: E731
    res.update(closest_contacts=contacts[:3], regio_sum_formed=dist(i, j) + dist(k, l),
               regio_sum_crossed=dist(i, l) + dist(k, j))
    checks = dict(
        flag_regio_ambiguous=res["regio_sum_crossed"] - res["regio_sum_formed"] < 0.3,   # D0: 1st pct 0.885 A, 4/5260 < 0
        shortest_contact_is_forming=contacts[0][1] in fset,
        one_imag=None if imag is None else len(imag) == 1,
        imag_below_min=None if not imag else imag[0] <= g["min_imag_cm"],
        no_foreign_bond=len(audit.foreign) == 0,
        forming_in_range=all(g["forming_min_A"] <= d for d in audit.d_formed),
        flag_async=max(audit.d_formed) > g["forming_max_A"],
        imag_mode_on_forming_bonds=None if share is None else share >= g["mode_share_min"],
    )
    res["checks"] = checks
    hard = [k for k in ("one_imag", "imag_below_min", "no_foreign_bond", "forming_in_range",
                        "imag_mode_on_forming_bonds") if checks[k] is False]
    if hard:
        raise StageFail("TS check failed: " + ",".join(hard))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--config"); ap.add_argument("--manifest")
    ap.add_argument("--dry-xtb", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    cfg = pc.load_config(a.config); job = pc.load_job(a.job, a.manifest); jd = pc.job_dir(cfg, a.job)
    if pc.done(jd, "ts"):
        print(f"{a.job}: ts already done"); return
    fr, _ = pc.import_repo_modules(cfg)
    res = dict(job_id=a.job, kind=job["kind"], ts_id=job["ts_id"], rxn_smiles=job["rxn_smiles"],
               engine=cfg["ts_stage"]["engine"], dry_xtb=a.dry_xtb)
    t0 = time.time()
    try:
        got = (load_replay(cfg, job, jd, res) if job["kind"] == "D0_replay"
               else run_autode(cfg, job, jd, res, a.dry_xtb))
        res["wall_s"] = time.time() - t0
        analyse(cfg, job, jd, res, got, fr)
    except StageFail as e:
        res["wall_s"] = time.time() - t0
        pc.write_json(jd / "ts_result.json", res)
        pc.mark_fail(jd, "ts", str(e)); return
    pc.write_json(jd / "ts_result.json", res)
    pc.mark_done(jd, "ts")
    im = res["imag_freqs_cm"]
    print(f"{a.job}: TS ok  imag={im[0] if im else 'n/a'}  d_form={['%.3f' % d for d in res['formed_d']]}  "
          f"dE‡(sp)={res.get('dE_act_sp_kcal')}  config TS/reactant={res.get('ts_dipole_config')}/"
          f"{res.get('reactant_dipole_config')}  wall={res['wall_s'] / 3600:.2f} h")


if __name__ == "__main__":
    main()
