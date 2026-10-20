#!/usr/bin/env python3
"""d1_split.py — D1PAIR Phase S (§4, §5): outer / inner splits, T2 families, leak audit, D0-D1 overlap, PREREG.

  python d1_split.py build      # results/S_folds.csv, S_report.json (exit 3 = STOP: T2 list / counts, leak audit)
  python d1_split.py prereg     # results/PREREG_D1PAIR.md + prereg_d1pair.json (commit them before any training)

Splits (fold = the held-out group of a pair; -1 = always on the training side):
  T1    core 5-fold: the 46 reference-pair cores sorted, numpy.random.default_rng(20261019).permutation, then each core
        into the fold with the fewest P_main pairs so far (ties: lower fold); anchor pairs follow their core; the 4
        anchor-only cores are always training (-1)
  LODO  fold = dipole (5);  LOSO  fold = skeleton (10, D'' excluded)
  T2    = T1 folds; the pairs of the 11 families with a DFT V call (appendix D calls on P_all, para) never train
Inner split (Phase M): GroupKFold(4), group = core, on the training-side T_main pairs of each outer fold.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1pair_common as C  # noqa: E402

FOLDS = C.RES / "S_folds.csv"
PREREG_MD = C.RES / "PREREG_D1PAIR.md"
PREREG_JSON = C.RES / "prereg_d1pair.json"


def t2_families():
    """appendix D on P_all DFT values: (families with any V call, hidden V, hidden bend, call table)."""
    import hammett_eval as HE
    P = pd.read_csv(C.PAIRS)
    rows = []
    for f in sorted(P[P.pos == "p"].fam.unique()):
        s = HE.family_series(P, f, "dd_barrier")
        if s is None:
            continue
        x, _, names = s
        Y = np.c_[[HE.family_series(P, f, f"dd_{c}")[1] for c in HE.CH]].T
        ok, v, *_ = HE.calls(x, Y, names)
        rows.append(dict(fam=f, **{f"bent_{c}": bool(ok[k]) for k, c in enumerate(HE.CH)},
                         **{f"V_{c}": bool(v[k]) for k, c in enumerate(HE.CH)}))
    T = pd.DataFrame(rows)
    vfam = sorted(T.loc[T[[f"V_{c}" for c in HE.CH]].any(axis=1), "fam"])
    ch = HE.CH[1:]
    hid_bend = sorted(T.loc[~T.bent_barrier & T[[f"bent_{c}" for c in ch]].any(axis=1), "fam"])
    hid_v = sorted(T.loc[~T.bent_barrier & T[[f"V_{c}" for c in ch]].any(axis=1), "fam"])
    return vfam, hid_v, hid_bend, T, P


def canon_rxn(smi):
    """reaction SMILES without atom maps and stereo: sorted canonical reactants >> canonical product."""
    from rdkit import Chem
    def can(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToSmiles(m)
    r, p = smi.split(">>")
    rs = sorted(can(x) for x in r.split("."))
    return ".".join(rs) + ">>" + str(can(p))


def cmd_build():
    T = pd.read_csv(C.RES / "F_pairs.csv")
    ref = T[T.ref_kind == "reference"]
    anc = T[T.ref_kind == "anchor"]
    fails = []
    # ---- T1
    all_ref_cores = sorted(ref.core.unique())
    if len(all_ref_cores) != C.EXPECT_P2["cores"]:
        fails.append(f"reference cores {len(all_ref_cores)}")
    # user decision 2026-10-19 (C.CORE_LOSS_DECISION): only cores that still have P_main reference pairs get a fold;
    # a core without any (azomethine ylide/A, carbonyl ylide/A) is anchor-only = always training (-1)
    cores = sorted(ref.loc[ref.in_P_main, "core"].unique())
    no_ref = sorted(set(all_ref_cores) - set(cores))
    if set(no_ref) - set(C.CORE_LOSS_ACCEPTED):
        fails.append(f"cores without P_main pairs not accepted by the user: {sorted(set(no_ref) - set(C.CORE_LOSS_ACCEPTED))}")
    perm = list(np.random.default_rng(C.SEED).permutation(np.array(cores, dtype=object)))
    nmain = ref[ref.in_P_main].groupby("core").size().reindex(cores).fillna(0).astype(int)
    load, fold_of = [0] * C.N_FOLDS, {}
    for c in perm:
        k = int(np.argmin(load))
        fold_of[c] = k
        load[k] += int(nmain[c])
    T["fold_T1"] = T.core.map(fold_of).fillna(-1).astype(int)
    # ---- LODO / LOSO
    dips = sorted(T.dipole.unique())
    skels = sorted(s for s in T.skeleton.unique() if s not in C.DPP)
    T["fold_LODO"] = T.dipole.map({d: k for k, d in enumerate(dips)}).astype(int)
    T["fold_LOSO"] = T.skeleton.map({s: k for k, s in enumerate(skels)}).astype(int)
    # ---- T2
    vfam, hid_v, hid_bend, calls, Pall = t2_families()
    n827 = int(Pall.fam.isin(vfam).sum())
    chk = dict(n_vfam=(len(vfam), len(C.T2_FAMILIES)), vfam_list=(vfam, sorted(C.T2_FAMILIES)),
               n_pairs_P_all=(n827, C.T2_N_PAIRS_PALL), hidden_V=(hid_v, sorted(C.T2_HIDDEN_V)),
               n_hidden_bend=(len(hid_bend), C.T2_N_HIDDEN_BEND))
    t2bad = {k: v for k, v in chk.items() if v[0] != v[1]}
    if t2bad:
        fails.append(f"T2 {sorted(t2bad)}")
    base_f = T.fam.str.replace("/anchor$", "", regex=True)
    T["t2_exclude"] = T.fam.isin(vfam) | ((T.ref_kind == "anchor") & base_f.isin(vfam))
    T["fold_T2"] = T.fold_T1
    T["hidden_V_fam"] = T.fam.isin(hid_v)
    T["hidden_bend_fam"] = T.fam.isin(hid_bend)
    # ---- T_main membership (training candidates) and evaluation sets
    T["in_T_main"] = T.in_P_main | T.in_A_main
    # ---- leak audit S-1
    audit = {}
    for sp in ("T1", "LODO", "LOSO", "T2"):
        col = f"fold_{sp}"
        for k in sorted(set(T[col]) - {-1}):
            te = T[(T[col] == k) & (T.in_P_sens | T.in_A_main)]
            tr = T[(T[col] != k) & T.in_T_main & ~(T.t2_exclude if sp == "T2" else False)]
            gcol = {"T1": "core", "T2": "core", "LODO": "dipole", "LOSO": "skeleton"}[sp]
            g_overlap = sorted(set(te[gcol]) & set(tr[gcol]))
            r_overlap = len((set(te.a_job) | set(te.b_job)) & (set(tr.a_job) | set(tr.b_job)))
            t2_leak = int(tr.t2_exclude.sum()) if sp == "T2" else 0
            audit[f"{sp}_{k}"] = dict(n_train=len(tr), n_test_P_main=int(te.in_P_main.sum()), n_test_A_main=int(te.in_A_main.sum()),
                                      group_overlap=g_overlap, reaction_overlap=r_overlap, t2_leak=t2_leak)
            if g_overlap or r_overlap or t2_leak:
                fails.append(f"leak {sp} fold {k}")
    # ---- same reactant molecule in several cores (report)
    ra = pd.read_pickle(C.ROWATTR).set_index("job_id")
    xmol = np.where(ref.series == "R1", ref.a_job.map(ra.dip_smi), ref.a_job.map(ra.dph_smi))
    core_of_mol = pd.DataFrame(dict(mol=xmol, core=ref.core.values)).groupby("mol").core.nunique()
    n_shared = int((pd.Series(xmol).map(core_of_mol) > 1).sum())
    # ---- D0 / D1 overlap (canonical reaction SMILES without maps / stereo)
    d0 = pd.read_csv(C.D0_SMILES)
    d0ok = set(int(r) for r in pd.read_csv(C.ESP_DIR / "results_rev6" / "S_splits_appendixA.csv").rxn_id.unique())
    d0c = {canon_rxn(s) for r, s in zip(d0.rxn_id, d0.rxn_smiles) if int(r) in d0ok}
    lab = {d["job_id"]: d for d in C.load_d1_labels()}
    used = sorted(set(T.a_job) | set(T.b_job))
    over = sorted(j for j in used if canon_rxn(lab[j]["rxn_smiles"]) in d0c)
    T["d0_overlap"] = T.a_job.isin(over) | T.b_job.isin(over)
    keep = ["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "core", "dipole", "skeleton", "series", "fam", "pos",
            "Xb", "sigma", "in_P_all", "in_P_feat", "in_P_main", "in_P_sens", "in_A_feat", "in_A_main", "in_T_main",
            "fold_T1", "fold_LODO", "fold_LOSO", "fold_T2", "t2_exclude", "hidden_V_fam", "hidden_bend_fam", "d0_overlap"]
    S = T[keep].sort_values("pair_id").reset_index(drop=True)
    C.write_atomic(FOLDS, lambda t: S.to_csv(t, index=False))
    rep = dict(n_pairs=len(S), n_fold_cores=len(cores), cores_without_P_main=no_ref, core_loss_user_decision=C.CORE_LOSS_DECISION,
               fold_cores={k: sorted(c for c, f in fold_of.items() if f == k) for k in range(C.N_FOLDS)},
               fold_P_main=load, perm_order=list(map(str, perm)), lodo_groups=dips, loso_groups=skels,
               t2_check={k: dict(got=v[0], want=v[1], ok=v[0] == v[1]) for k, v in chk.items()},
               t2_exclude_pairs=dict(P_all=int((S.ref_kind.eq("reference") & S.t2_exclude).sum()),
                                     P_main=int((S.in_P_main & S.t2_exclude).sum()),
                                     anchor_same_F=int((S.ref_kind.eq("anchor") & S.t2_exclude).sum())),
               hidden_bend_families=hid_bend, hidden_V_families=hid_v,
               appendix_D_calls=dict(bent={c: int(calls[f"bent_{c}"].sum()) for c in ("barrier", "d1", "d2", "elst", "pauli",
                                                                                  "oi", "disp", "cpcm", "cds")},
                                     V={c: int(calls[f"V_{c}"].sum()) for c in ("barrier", "d1", "d2", "elst", "pauli",
                                                                             "oi", "disp", "cpcm", "cds")}),
               leak_audit=audit, same_X_molecule_in_other_core=dict(n_pairs=n_shared, of=len(ref)),
               d0_overlap=dict(n_d1_reactions=len(over), reactions=over[:50], n_pairs=int(S.d0_overlap.sum()),
                               n_P_main=int((S.d0_overlap & S.in_P_main).sum())),
               counts=dict(P_all=int(S.in_P_all.sum()), P_feat=int(S.in_P_feat.sum()), P_main=int(S.in_P_main.sum()),
                           P_sens=int(S.in_P_sens.sum()), A_feat=int(S.in_A_feat.sum()), A_main=int(S.in_A_main.sum()),
                           T_main=int(S.in_T_main.sum())),
               fails=fails, stop=bool(fails), folds_sha256=C.sha256(FOLDS))
    C.write_json_safe(C.RES / "S_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "leak_audit"}, indent=1, default=str))
    return 3 if fails else 0


# ================================================================ PREREG
def prereg_record():
    S = pd.read_csv(FOLDS)
    srep = json.loads((C.RES / "S_report.json").read_text())
    f0 = json.loads((C.RES / "f0_report.json").read_text())

    def lsha(m):
        return C.list_sha(S.loc[m, ["pair_id", "a_job", "b_job"]].itertuples(index=False))
    inputs = [C.D1_LABELS, C.D1_STATUS, C.D0_LABELS, Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/xtb_r6/X182_all.parquet"),
              Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/S/S_targets.parquet"),
              C.ESP_DIR / "results_rev6" / "S_splits_appendixA.csv", C.X182_D1, C.RES / "f0_d1.parquet",
              C.RES / "F_pairs.csv", C.PAIRS, C.ANCHOR, C.RES / "pairs_d1_anchor_feat.csv", C.P4_DROP, FOLDS, C.ROWATTR,
              C.RES / "f0_report.json"]
    import train_ml_single as T
    grids = {"Ridge": {"alpha": C.grid_values("Ridge")["alpha"]},
             "KRR": {"alpha": C.grid_values("KRR_rbf")["alpha"], "gamma_delta": C.GAMMAS, "gamma_xbar": C.GAMMAS},
             "SVR": {"C": C.SVR_C, "epsilon": C.SVR_EPS, "gamma_delta": C.GAMMAS, "gamma_xbar": C.GAMMAS},
             "XGB": C.grid_values("XGB"), "NN2/NN4": "r6_nn.HP_STORAGE + r6_nn.HB (Hyperband max_epochs 100, factor 3, seed 23)",
             "P (f_pool KRR)": {k.split("__")[-1]: v for k, v in T.GRIDS["KRR_rbf"][1].items()}}
    return dict(
        spec="docs/specs/D1PAIR_REV9.md", date=time.strftime("%Y-%m-%d"), constants_source="analysis/d1_pair/d1pair_common.py",
        inputs_sha256={str(p): C.sha256(p) for p in inputs},
        pair_lists_sha256=dict(P_main=lsha(S.in_P_main), P_sens=lsha(S.in_P_sens), A_main=lsha(S.in_A_main),
                               T_main=lsha(S.in_T_main)),
        counts=srep["counts"], s_folds_sha256=C.sha256(FOLDS), t2_families=srep["t2_check"]["vfam_list"]["got"],
        t2_hidden_V=srep["hidden_V_families"], t2_hidden_bend=srep["hidden_bend_families"],
        f0_representative=f0["representative"], f0_gate=f0["gate_F0_1"],
        arms={"null": "0", "A": "f0(x_a) - f0(x_b) (f0 representative)", "B": "f0 diff + g, g on dd - f0 diff, T_main",
              "C": "g on dd, T_main", "P": "KRR f_pool on D0 4,963 + training-side D1 reactions, f_pool(x_a) - f_pool(x_b)",
              "B_ref": "B with g = KRR on training-side P_main only"},
        units={"T1/LODO/LOSO": "arms B, C x 6 models x 10 targets x outer folds", "T2": "arm B x 6 models x 10 targets x 5 folds",
               "P, B_ref": "KRR x 10 targets x T1 5 folds"},
        models=C.MODELS, targets=C.TARGETS, splits=list(C.SPLITS), grids=grids,
        scale_rule="z = [dx / s_delta, (xbar - mu) / s] on the (inner) training pairs, no mean removed from dx or y; y / s_y; "
                   "columns with s < 1e-12 dropped and recorded",
        antisymmetry="Ridge on dx only, no intercept; KRR / SVR antisymmetric precomputed kernel "
                     "[k(dx_p, dx_q) - k(dx_p, -dx_q)] k(xbar_p, xbar_q); SVR / XGB trained both ways, predict "
                     "1/2 [s(dx) - s(-dx)]; NN g = h([dx, xbar]) - h([-dx, xbar]) with shared h (r6_nn._layers); "
                     "gate M-1 |g(dx) + g(-dx)| <= 1e-6 and |g(0)| <= 1e-6 kcal/mol on 200 random test pairs",
        tuning="GroupKFold(4), group = core, on the training-side T_main pairs; NN: Hyperband on inner fold 1, CV MAE = "
               "mean over the 4 inner folds refitted with the best hps (500 epochs)",
        representative_rule="per split x arm x target: lowest mean (over outer folds) inner-CV MAE among the 6 models; "
                            "ties by model order; never by test score",
        metrics="test pairs of all outer folds pooled; primary = P_main reference pairs; anchor test MAE separately",
        stages={"0": "MAE(B) < MAE(null) and MAE(A) - MAE(B) 95 % CI low > 0 and Holm p < 0.05 (core cluster bootstrap "
                     "2,000, seed 20261019)",
                "1": f"median R2_fam of the test para families >= {C.STAGE1_R2}",
                "2": f"oi and cpcm para-pair MAE <= {C.STAGE2_MAE}, bend-call sensitivity >= {C.STAGE2_SENS}, false positive "
                     f"rate <= {C.STAGE2_FPR}"},
        primary_tests="B representative vs A per target in " + ", ".join(C.PRIMARY_TARGETS) + "; metric = test-pair MAE "
                      "difference A - B; 95 % CI and two-sided p by core cluster bootstrap (2,000, seed 20261019, the cores "
                      "of the T1 test pairs (44 after the user decision) resampled with replacement); Holm over the 9; "
                      "folds where B is lower counted",
        secondary="B vs C, B vs null, B vs P, B(KRR) vs B_ref(KRR), every model pair, sum-of-9 barrier vs direct, "
                  "LODO / LOSO, P_sens, without D0-overlap reactions, T2 vs T1 (hidden V 6, hidden bend 30; "
                  "Clopper-Pearson 95 % CI for sensitivities)",
        user_decisions=[C.P4_DECISION, C.SI_DECISION, C.F0_DECISION, C.CORE_LOSS_DECISION, C.AFEAT_DECISION,
                        "D1_JOBS extracted from the D1 archives (2026-10-19)",
                        "G1 rows whose ORCA died at MPI start-up rerun with the unchanged first attempt (results/G_infra_retry.csv)"],
        rounding={"mae": 2, "mae_fine (disp, cds, c_ghost)": 3, "r2": 3, "p_sig": 2})


def cmd_prereg():
    rec = prereg_record()
    C.write_json_safe(PREREG_JSON, rec)
    md = ["# PREREG — D1PAIR rev 9 (written before any g training)", "",
          f"Spec `docs/specs/D1PAIR_REV9.md`; constants `analysis/d1_pair/d1pair_common.py`; machine-readable record "
          f"`prereg_d1pair.json` (inputs sha256, pair-list sha256, S_folds sha256, grids). Date {rec['date']}.", "",
          "## Data", "", *[f"- {k}: {v}" for k, v in rec["counts"].items()], "",
          f"- f0 representative per target: {rec['f0_representative']}", "",
          "## Arms", "", *[f"- **{k}**: {v}" for k, v in rec["arms"].items()], "",
          "## Units", "", *[f"- {k}: {v}" for k, v in rec["units"].items()], "",
          "## Rules", "", f"- scaling: {rec['scale_rule']}", f"- antisymmetry: {rec['antisymmetry']}",
          f"- tuning: {rec['tuning']}", f"- representative model: {rec['representative_rule']}",
          f"- metrics: {rec['metrics']}", "", "## Stage criteria (arm B representative, T1 test pairs)", "",
          *[f"- stage {k}: {v}" for k, v in rec["stages"].items()], "", "## Tests", "",
          f"- primary (confirmatory, 9): {rec['primary_tests']}", f"- secondary (descriptive, uncorrected): {rec['secondary']}",
          "", "## T2 (model that never saw a V)", "", f"- excluded families: {rec['t2_families']}",
          f"- hidden V: {rec['t2_hidden_V']}", f"- hidden bend: {len(rec['t2_hidden_bend'])} families", "",
          "## User decisions recorded before training", "", *[f"- {d}" for d in rec["user_decisions"]], "",
          "Any later change to arms, models, pairs, metrics or criteria is reported as a deviation in SUMMARY.md."]
    C.write_atomic(PREREG_MD, "\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def gate_prereg():
    """training gate (rev6_common.gate_prereg style): PREREG files committed + unchanged and every input sha256 equal."""
    def git(*a):
        return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (PREREG_MD, PREREG_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = json.loads(PREREG_JSON.read_text())
        for path, sha in rec["inputs_sha256"].items():
            if not Path(path).is_file():
                bad.append(f"input {path} missing")
            elif C.sha256(path) != sha:
                bad.append(f"input {path} changed")
    if bad:
        print("PREREG gate (§5: commit PREREG_D1PAIR before any training):\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(3)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())




ADD1_MD = C.RES / "PREREG_D1PAIR_ADD1.md"
ADD1_JSON = C.RES / "prereg_d1pair_add1.json"


def gate_add1():
    """FU1 Phase A gate (same method as gate_prereg): PREREG_D1PAIR_ADD1.md / prereg_d1pair_add1.json committed and
    unchanged since the commit, and their content sha256 equal to the values recorded in the json; exits 3 otherwise."""
    def git(*a):
        return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (ADD1_MD, ADD1_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec = {}
    if not bad:
        rec = json.loads(ADD1_JSON.read_text())
        if C.sha256(ADD1_MD) != rec.get("md_sha256"):
            bad.append("PREREG_D1PAIR_ADD1.md sha256 differs from prereg_d1pair_add1.json md_sha256")
    if bad:
        print("PREREG ADD1 gate (FU1 Phase A: commit the addendum before L / X / LODO-LOSO evaluation):\n  " + "\n  ".join(bad),
              file=sys.stderr)
        sys.exit(3)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"build": cmd_build, "prereg": cmd_prereg}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn())
