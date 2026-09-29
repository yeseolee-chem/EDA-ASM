#!/usr/bin/env python3
"""sample_pilot.py — draw the D1 pilot set from D1_설계_v8.xlsx (sheet 9 = one TS per row).

Default: stratified random, 2 TS rows per design panel (5 panels x 2 = 10), backbone
substituents only (no '복제' role), fixed seed. One job per TS row, with the workbook's
autodE SMILES unchanged (policy "미지정 — autodE 최저 입체 선택 (Coley 규약)"): autodE builds
an addition TS from the product side, so an E/Z tag on the reactant dipole would not fix
the TS configuration (autodE 1.4.5 reaction.py::locate_transition_state); the stereo-
compatible reference is chosen afterwards (make_reference.py).
--force-stereo guarantees that at least one sampled row has a dipole with an unspecified
C=X bond from --stereo-cores, so the reference rule is exercised on the A grid.
D0 controls (Coley reactions in labels_all.json): one with alt_used=1 (Coley replaced the
dipole reference) and one with alt_used=0, each run twice: D0_control (autodE from SMILES,
full pipeline) and D0_replay (Coley's TS and reactants copied in; later stages only).

usage:
  python sample_pilot.py --xlsx D1_설계_v8.xlsx --out pilot_manifest.csv \
      [--seed 20260929] [--mode stratified|pure] [--n 10] [--force-stereo 1] \
      [--controls 2 --coley-csv full_dataset.csv --labels labels_all.json]
"""
import argparse, json, random, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rdkit import Chem

PANELS = {"A": "P1_A_grid", "B": "P2_B_aryl", "C_lo": "P3_C1-6", "C_hi": "P4_C7-10", "D": "P5_D"}


def panel_of(core: str) -> str:
    if core.startswith("A"): return PANELS["A"]
    if core.startswith("B"): return PANELS["B"]
    if core.startswith("C"): return PANELS["C_lo"] if int(core[1:]) <= 6 else PANELS["C_hi"]
    if core.startswith("D"): return PANELS["D"]
    raise ValueError(core)


def load_rows(xlsx: str) -> pd.DataFrame:
    cells = pd.read_excel(xlsx, sheet_name="4_계산목록", header=3)
    cells = cells[cells["셀ID"].astype(str).str.fullmatch(r"D1-\d+")]
    ts = pd.read_excel(xlsx, sheet_name="9_autodE입력", header=3)
    ts = ts[ts["TS_ID"].astype(str).str.fullmatch(r"D1-\d+-r\d")]
    backbone = cells[~cells["R1 역할"].eq("복제") & ~cells["R2 역할"].eq("복제")]
    ts = ts[ts["셀ID"].isin(backbone["셀ID"])].merge(
        backbone[["셀ID", "R1", "R2", "dipole SMILES (전체)", "dipolarophile SMILES (전체)"]], on="셀ID", how="left")
    ts["panel"] = ts["코어"].map(panel_of)
    return ts.reset_index(drop=True)


def unspecified_db(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    return sum(1 for s in Chem.FindPotentialStereo(m)
               if s.type == Chem.StereoType.Bond_Double and s.specified == Chem.StereoSpecified.Unspecified)


def dipole_of(mapped_rxn: str):
    """(dipole SMILES, dipolarophile SMILES), maps removed; dipole = 3 atoms in the new ring."""
    from stereo_utils import dipole_backbone
    k, _, _ = dipole_backbone(mapped_rxn)
    parts = mapped_rxn.split(">>")[0].split(".")
    clean = []
    for p in parts:
        m = Chem.MolFromSmiles(p)
        for at in m.GetAtoms():
            at.SetAtomMapNum(0)
        clean.append(Chem.MolToSmiles(m))
    return clean[k], clean[1 - k]


def strip_maps(rxn: str) -> str:
    """autodE needs map-free SMILES (Coley ReactionDataPoint.process_smiles)."""
    r, p = rxn.split(">>")
    out = []
    for side in (r, p):
        m = Chem.MolFromSmiles(side)
        for at in m.GetAtoms():
            at.SetAtomMapNum(0)
        out.append(Chem.MolToSmiles(m))
    return ">>".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--mode", choices=["stratified", "pure"], default="stratified")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--force-stereo", type=int, default=1,
                    help="guarantee at least this many rows from --stereo-cores whose dipole C=X is unspecified")
    ap.add_argument("--stereo-cores", default="A1,A3")
    ap.add_argument("--controls", type=int, default=2)
    ap.add_argument("--coley-csv"); ap.add_argument("--labels")
    ap.add_argument("--max-jobs", type=int, default=15, help="SLURM MaxSubmit is 20 tasks: jobs + 1 report task must fit with headroom")
    ap.add_argument("--exclude-ts", default="",
                    help="TS_IDs to leave out of the pool: comma list, or a CSV with a ts_id column (validation V6a)")
    ap.add_argument("--id-prefix", default="J", help="job_id prefix (validation V6a uses V6a_)")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    ts = load_rows(a.xlsx)
    if a.exclude_ts:
        p = Path(a.exclude_ts)
        excl = set(pd.read_csv(p)["ts_id"].astype(str)) if p.is_file() else set(a.exclude_ts.split(","))
        ts = ts[~ts["TS_ID"].astype(str).isin(excl)].reset_index(drop=True)
        print(f"excluded {len(excl)} TS ids", file=sys.stderr)
    ts["n_unspec"] = ts["dipole SMILES (전체)"].map(unspecified_db)
    print(f"backbone pool: {ts['셀ID'].nunique()} cells / {len(ts)} TS rows", file=sys.stderr)
    print(ts.groupby("panel").size().to_string(), file=sys.stderr)

    if a.mode == "pure":
        pick = rng.sample(list(ts.index), a.n)
    else:
        per = a.n // len(PANELS)
        pick = []
        for p in sorted(ts.panel.unique()):
            pick += rng.sample(list(ts.index[ts.panel == p]), per)
    chosen = ts.loc[pick].copy()
    # guarantee stereo coverage: swap one panel-A pick for an eligible row if none was drawn
    st_cores = a.stereo_cores.split(",")
    need = a.force_stereo - int(((chosen.n_unspec > 0) & chosen["코어"].isin(st_cores)).sum())
    if need > 0:
        elig = [i for i in ts.index[(ts.n_unspec > 0) & ts["코어"].isin(st_cores)] if i not in pick]
        swap_out = [i for i in chosen.index[chosen.panel == PANELS["A"]]][:need]
        swap_in = rng.sample(elig, len(swap_out))
        chosen = pd.concat([chosen.drop(swap_out), ts.loc[swap_in]])
        print(f"force-stereo: replaced {swap_out} with {swap_in}", file=sys.stderr)

    jobs = []
    for _, r in chosen.iterrows():
        base = dict(kind="D1", ts_id=r["TS_ID"], cell_id=r["셀ID"], core=r["코어"], panel=r["panel"],
                    regio=r["regio"], R1=r["R1"], R2=r["R2"], charge=int(r["순전하"]), mult=int(r["다중도"]),
                    product_smiles=r["product SMILES"], mapped_smiles=r["rxn SMILES (atom-mapped, Coley 규약)"],
                    dipole_smiles=r["dipole SMILES (전체)"], dipolarophile_smiles=r["dipolarophile SMILES (전체)"],
                    n_unspec_dipole_db=int(r["n_unspec"]), rxn_smiles=r["rxn SMILES (autodE 입력, 맵 제거)"])
        jobs.append(base)

    if a.controls:
        coley = pd.read_csv(a.coley_csv)
        lab = {int(x["rxn_id"]): x for x in json.load(open(a.labels)) if x["status"] == "ok"}
        ok = coley[coley.rxn_id.isin(lab)].copy()
        ok = ok[[lab[i]["charge1"] == 0 and lab[i]["charge2"] == 0 and lab[i]["n_atoms"] <= 30
                 and not lab[i]["flag_async"] for i in ok.rxn_id]]
        ok["alt"] = [lab[i]["alt_used"] for i in ok.rxn_id]
        ok["dip_unspec"] = [unspecified_db(dipole_of(s)[0]) for s in ok.rxn_smiles]
        # alt_used=1 control from dipoles with a stereogenic C=X (the case the reference rule is for)
        picks = [ok[(ok.alt == 1) & (ok.dip_unspec > 0)].sample(1, random_state=a.seed),
                 ok[ok.alt == 0].sample(1, random_state=a.seed)][:a.controls]
        CH = ("barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft",
              "cds_dft", "formed_d1", "formed_d2", "alt_used", "rel1_file", "rel2_file")
        for kind in ("D0_control", "D0_replay"):
            for _, c in pd.concat(picks).iterrows():
                L = lab[int(c.rxn_id)]
                dsmi, psmi = dipole_of(c.rxn_smiles)
                jobs.append(dict(kind=kind, ts_id=f"D0-{int(c.rxn_id):04d}", cell_id="", core="D0",
                                 panel=kind, regio="", R1="", R2="", charge=0, mult=1,
                                 product_smiles=strip_maps(c.rxn_smiles).split(">>")[1], mapped_smiles=c.rxn_smiles,
                                 rxn_smiles=strip_maps(c.rxn_smiles), dipole_smiles=dsmi, dipolarophile_smiles=psmi,
                                 n_unspec_dipole_db=int(c.dip_unspec), d0_rxn_id=int(c.rxn_id),
                                 d0_G_act_kcal=float(c.G_act), **{f"d0_{k}": L[k] for k in CH}))

    out = pd.DataFrame(jobs)
    out.insert(0, "job_id", [f"{a.id_prefix}{i:02d}" for i in range(len(out))])
    if len(out) > a.max_jobs:
        sys.exit(f"STOP: {len(out)} jobs > --max-jobs {a.max_jobs} (SLURM MaxSubmit=20). Lower --n or --controls.")
    out.to_csv(a.out, index=False)
    print(out[["job_id", "kind", "ts_id", "core", "panel", "regio", "n_unspec_dipole_db", "rxn_smiles"]].to_string(index=False))


if __name__ == "__main__":
    main()
