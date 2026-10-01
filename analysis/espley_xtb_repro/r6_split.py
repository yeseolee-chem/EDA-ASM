#!/usr/bin/env python3
"""r6_split.py — rev 6 Phase S (docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md §2, §7): fix rows, splits and the inputs
of every training before PREREG_REV6 is committed.

  python r6_split.py build      # -> results_rev6/S_rows.csv, S_splits.csv, S_splits_appendixA.csv,
                                #    S_label_checks.json, S.json; $R6_SCRATCH/S/S_targets.parquet, S_X_esp.parquet

Rows (ds3 order = Espley's manual_tt_solvent.pkl row order, position 0..3509):
  R_esp all 3,510; R_lab label ok; R_feat our 182 features complete (R_rows_final.csv from r6_features.py report);
  R_hyg d1 >= 0, d2 >= 0, d2 <= 50; C = R_lab ∩ R_feat ∩ R_hyg; C_all = R_lab ∩ R_feat;
  D = (R_lab ∩ R_hyg) \\ R_feat (the rows our side lost: bias check, §2-2).
Split per seed (22, 23, 14, 1, 2): rev6_common.espley_split3 on positions 0..3509 = Espley's
_perform_train_test_split (train 80 %, test = first half of the 20 % hold-out, val = second half), asserted equal to
compare_espley.espley_split; train_s / test_s / val_s = that part ∩ C (order kept), likewise for C_all and D.
Appendix A rows: R_lab ∩ R_feat ∩ R_hyg over all labels, rxn_id ascending, train_ml_single.split_80_10_10.
Targets: labels_all.json values (dft_* names of rev5_common.TARGETS9; c_ghost = barrier - d1 - d2 - e_bond as
aggregate.derive). Espley X: their 46 fit + 1 tuning column (sum_distortion_energies_am1), the two AM1 distortions
re-ordered to dipole / dipolarophile by results_rev6/P3_swap.csv (rev 4 swap vector).
Every output is written atomically; rerunning rewrites identical files (the sha256 values in S.json must not change).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev6_common as R6  # noqa: E402

R5 = R6.R5
LABEL_KEY = {"dft_d1_kcal": "d1_kcal", "dft_d2_kcal": "d2_kcal", "dft_eint_spe_kcal": "eint_spe_kcal",
             "dft_barrier_kcal": "barrier_kcal", "dft_e_bond_kcal": "e_bond_kcal", "dft_elst_dft": "elst_dft",
             "dft_pauli_dft": "pauli_dft", "dft_oi_dft": "oi_dft", "dft_disp_dft": "disp_dft",
             "dft_cpcm_dft": "cpcm_dft", "dft_cds_dft": "cds_dft", "bsse_shift_kcal": "bsse_shift_kcal"}
ESP_DROP_ML = ("contributions", "reactant", "structure", "path", "reaction_number", "sum_distortion_energies")


def ids_sha(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def espley_x_columns(cols, drop):
    """Their X: drop tag columns (ml_analysis.py :139-145 / hyp_tuning.py :100-106), then every '_dft' column."""
    return [c for c in cols if not any(t in c for t in drop) and "_dft" not in c]


def build():
    import compare_espley as CE
    import train_ml_single as T
    rows = pd.read_csv(R6.ROWS_R)
    esp = pd.read_pickle(R6.ESP_DS).reset_index(drop=True)
    ids = esp["reaction_number"].astype(int).to_numpy()
    assert len(ids) == R6.N_ESP and len(set(ids)) == R6.N_ESP
    r = rows.set_index("rxn_id")
    S = pd.DataFrame(dict(pos=np.arange(R6.N_ESP), rxn_id=ids))
    for c in ("R_lab", "R_feat", "R_hyg", "C", "C_all"):
        S[c] = [bool(r.at[i, c]) if i in r.index else False for i in ids]
    S["recovered_by"] = [r.at[i, "recovered_by"] if i in r.index else None for i in ids]
    S["not_feat_reason"] = [r.at[i, "not_feat_reason"] if i in r.index else "not_in_labels" for i in ids]
    S["D"] = S.R_lab & S.R_hyg & ~S.R_feat
    assert (S.C == (S.R_lab & S.R_feat & S.R_hyg)).all() and (S.C_all == (S.R_lab & S.R_feat)).all()
    R6.write_atomic(R6.RES6 / "S_rows.csv", lambda t: S.to_csv(t, index=False))

    spl, shas = [], {}
    inC, inCa, inD = S.C.to_numpy(), S.C_all.to_numpy(), S.D.to_numpy()
    for s in R6.SEEDS:
        tr, te, va = R6.espley_split3(R6.N_ESP, s)
        tr2, te2 = CE.espley_split(R6.N_ESP, s)
        assert np.array_equal(tr, tr2) and np.array_equal(te, te2), f"split mismatch seed {s}"
        for part, arr in (("train", tr), ("test", te), ("val", va)):
            for k, p in enumerate(arr):
                spl.append(dict(seed=s, part=part, order=k, pos=int(p), rxn_id=int(ids[p]), C=bool(inC[p]),
                                C_all=bool(inCa[p]), D=bool(inD[p])))
            for nm, m in (("C", inC), ("C_all", inCa), ("D", inD)):
                sel = [int(ids[p]) for p in arr if m[p]]
                shas[f"{s}/{part}/{nm}"] = dict(n=len(sel), sha256=ids_sha(sel))
    spl = pd.DataFrame(spl)
    R6.write_atomic(R6.RES6 / "S_splits.csv", lambda t: spl.to_csv(t, index=False))

    # targets (all labels with status ok)
    lab = [d for d in json.loads(Path(R5.LABELS).read_text()) if d.get("status") == "ok"]
    tg = pd.DataFrame([{"rxn_id": int(d["rxn_id"]), **{k: float(d[v]) for k, v in LABEL_KEY.items()}} for d in lab])
    tg["dft_c_ghost_kcal"] = tg.dft_barrier_kcal - (tg.dft_d1_kcal + tg.dft_d2_kcal + tg.dft_e_bond_kcal)
    tg = tg.sort_values("rxn_id").reset_index(drop=True)
    R6.write_atomic(R6.S_TARGETS, lambda t: tg.to_parquet(t, index=False))

    # Espley X (47 = 46 + the tuning-only sum column), role order
    sw = pd.read_csv(R6.RES6 / "P3_swap.csv")
    assert np.array_equal(sw.rxn_id.to_numpy(), ids)
    swap = sw.swap.to_numpy(float)
    x47 = espley_x_columns(esp.columns, ESP_DROP_ML[:-1])
    x46 = espley_x_columns(esp.columns, ESP_DROP_ML)
    assert len(x46) == 46 and [c for c in x47 if c not in x46] == ["sum_distortion_energies_am1"], (len(x46), x47)
    X = esp[x47].copy()
    a1, a2 = X["distortion_energy_1_am1"].to_numpy().copy(), X["distortion_energy_2_am1"].to_numpy().copy()
    X["distortion_energy_1_am1"], X["distortion_energy_2_am1"] = np.where(swap == 1, a2, a1), np.where(swap == 1, a1, a2)
    X.insert(0, "rxn_id", ids)
    R6.write_atomic(R6.S_X_ESP, lambda t: X.to_parquet(t, index=False))

    # our 182 features
    x182 = pd.read_parquet(R6.X182_ALL)
    cols = R6.our_columns(T.FEATURE_SETS)
    assert list(x182.columns) == ["rxn_id"] + cols and not x182[cols].isna().any().any()
    bad = [c for c in cols if any(t in c for t in ESP_DROP_ML) or "_dft" in c]
    assert not bad, f"our feature names collide with Espley's drop tags: {bad}"
    assert set(S.loc[S.C_all, "rxn_id"]) <= set(x182.rxn_id)

    # label relations on C (spec §7-2)
    t = tg.set_index("rxn_id").loc[S.loc[S.C, "rxn_id"]]
    six = t[[f"dft_{c}_dft" for c in R6.SIX]].sum(axis=1)
    chk = dict(n=len(t),
               max_abs_d1_d2_eint_minus_barrier=float((t.dft_d1_kcal + t.dft_d2_kcal + t.dft_eint_spe_kcal
                                                       - t.dft_barrier_kcal).abs().max()),
               max_abs_sum6_minus_e_bond=float((six - t.dft_e_bond_kcal).abs().max()),
               max_abs_eint_minus_e_bond_minus_c_ghost=float((t.dft_eint_spe_kcal - t.dft_e_bond_kcal
                                                              - t.dft_c_ghost_kcal).abs().max()),
               max_abs_eint_minus_e_bond_minus_bsse_shift=float((t.dft_eint_spe_kcal - t.dft_e_bond_kcal
                                                                 - t.bsse_shift_kcal).abs().max()),
               c_ghost_mean=float(t.dft_c_ghost_kcal.mean()), c_ghost_sd=float(t.dft_c_ghost_kcal.std()),
               sum6_minus_eint_mean=float((six - t.dft_eint_spe_kcal).mean()))
    R6.write_json(R6.RES6 / "S_label_checks.json", chk)

    # appendix A (Protocol A, all labels)
    allr = rows[rows.R_lab & rows.R_feat & rows.R_hyg].sort_values("rxn_id").rxn_id.to_numpy()
    sa = []
    from sklearn.model_selection import train_test_split
    for s in R6.SEEDS:
        tr, rest = train_test_split(np.arange(len(allr)), test_size=0.20, random_state=s)
        va, te = train_test_split(rest, test_size=0.50, random_state=s)     # Protocol A test = SECOND half
        tr2, te2 = T.split_80_10_10(len(allr), s)
        assert np.array_equal(tr, tr2) and np.array_equal(te, te2), f"appendix split mismatch seed {s}"
        for part, arr in (("train", tr), ("val", va), ("test", te)):
            sa += [dict(seed=s, part=part, order=k, rxn_id=int(allr[p])) for k, p in enumerate(arr)]
            shas[f"A/{s}/{part}"] = dict(n=len(arr), sha256=ids_sha(allr[arr]))
    sa = pd.DataFrame(sa)
    R6.write_atomic(R6.S_SPLITS_A, lambda t: sa.to_csv(t, index=False))

    rep = dict(counts=dict(R_esp=int(len(S)), R_lab=int(S.R_lab.sum()), R_feat=int((S.R_lab & S.R_feat).sum()),
                           C=int(S.C.sum()), C_all=int(S.C_all.sum()), D=int(S.D.sum()), appendixA=int(len(allr))),
               split_sha256=shas, label_checks=chk,
               files={str(p.relative_to(R6.RES6)) if str(p).startswith(str(R6.RES6)) else str(p): R6.sha256(p)
                      for p in (R6.RES6 / "S_rows.csv", R6.RES6 / "S_splits.csv", R6.S_SPLITS_A, R6.S_TARGETS,
                                R6.S_X_ESP, R6.X182_ALL, R6.ROWS_R, R6.RES6 / "P3_swap.csv", R5.LABELS)},
               feature_cols_182=cols, esp_cols_46=x46, esp_cols_47=x47)
    R6.write_json(R6.RES6 / "S.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k in ("counts", "label_checks")}, indent=1))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "build":
        raise SystemExit(__doc__)
    sys.exit(build())
