#!/usr/bin/env python3
"""Validate stereo_utils on the D0 (Coley) geometries.

For every D0 reaction with status ok: extract the dipole from the TS with the repo's
graph partition, then compare the backbone configuration tags of
  TS dipole  vs  plain reactant dipole (r*.xyz)  vs  Coley's replacement (r*_alt.xyz).
Expected if the tags mean what Coley's _alt step fixes:
  (1) whenever _alt exists it is compatible with the TS            (should be ~100 %)
  (2) plain-vs-TS incompatibility occurs mostly where _alt exists
usage: python validate_config_on_coley.py --profiles DIR --csv full_dataset.csv \
         --labels labels_all.json --repo REPO [--n 0 (all)] [--out cfg_validation.csv]
"""
import argparse, json, sys, random
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import stereo_utils as su  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True); ap.add_argument("--csv", required=True)
    ap.add_argument("--labels", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="cfg_validation.csv")
    ap.add_argument("--min-alt-ok", type=float, default=0.90,
                    help="PASS needs this share of Coley _alt references to match the TS configuration")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.repo) / "analysis/espley_xtb_repro"))
    import fragmenter as fr

    smi = dict(pd.read_csv(a.csv)[["rxn_id", "rxn_smiles"]].values)
    lab = [r for r in json.load(open(a.labels)) if r["status"] == "ok"]
    if a.n:
        lab = random.Random(a.seed).sample(lab, a.n)
    rows = []
    for r in lab:
        rid = int(r["rxn_id"]); pdir = Path(a.profiles) / str(rid)
        rec = dict(rxn_id=rid, alt_used=r["alt_used"])
        try:
            _, bb, dip_smi = su.dipole_backbone(smi[rid])
            q = r["charge1"]                                   # frag1 = dipole in labels_all
            ts = fr.read_xyz(next(p for p in pdir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz"))
            plain = r["rel1_file"].replace("_alt", "")
            r_dip = fr.read_xyz(pdir / plain)
            others = [p for p in pdir.glob("r*.xyz") if "_alt" not in p.name and p.name != plain]
            r_dph = fr.read_xyz(others[0])
            P = fr.partition(ts, r_dip, r_dph)
            if P.status != "ok":
                rec["err"] = f"partition:{P.status}"; rows.append(rec); continue
            A = sorted(P.A)
            ts_blk = su.xyz_block_from_atoms([ts[0][i] for i in A], ts[1][A])
            rec["n_track"] = len(su.tracked_dihedrals(dip_smi, bb))
            rec["ts"], rec["ts_dih"], _ = su.config_tags(ts_blk, dip_smi, bb, q)
            rec["plain"], _, _ = su.config_tags(su.xyz_block_from_atoms(*r_dip), dip_smi, bb, q)
            altp = pdir / (plain[:-4] + "_alt.xyz")
            rec["has_alt"] = altp.is_file()
            if rec["has_alt"]:
                rec["alt"], _, _ = su.config_tags(su.xyz_block_from_atoms(*fr.read_xyz(altp)), dip_smi, bb, q)
            rec["plain_ok"] = su.tags_compatible(rec["plain"], rec["ts"])
            rec["alt_ok"] = su.tags_compatible(rec.get("alt", ""), rec["ts"]) if rec["has_alt"] else None
        except Exception as e:                                  # noqa: BLE001
            rec["err"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    ok = df[df.get("err").isna()] if "err" in df else df
    print(f"reactions: {len(df)}   errors: {df['err'].notna().sum() if 'err' in df else 0}")
    if "err" in df:
        print(df["err"].dropna().str.split(":").str[0].value_counts().to_string())
    trk = ok[ok.n_track > 0]
    print(f"with tracked dihedrals: {len(trk)} / {len(ok)}")
    print("plain reactant compatible with TS:", trk.plain_ok.value_counts().to_dict())
    print("_alt present:", trk.has_alt.value_counts().to_dict())
    print(pd.crosstab(trk.plain_ok, trk.has_alt, rownames=["plain_ok"], colnames=["has_alt"]).to_string())
    a2 = trk[trk.has_alt]
    print("_alt compatible with TS:", a2.alt_ok.value_counts().to_dict())
    und = (ok["ts"].fillna("").str.contains(r"\?")).sum()
    print(f"TS tags with an undefined ('?') dihedral: {und}")
    n_err = int(df["err"].notna().sum()) if "err" in df else 0
    share = float(a2.alt_ok.mean()) if len(a2) else 0.0
    ok_all = n_err == 0 and share >= a.min_alt_ok
    print(f"CONFIG_VALIDATION {'PASS' if ok_all else 'FAIL'} (errors {n_err}, _alt compatible {share:.3f})")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
