"""Equivalence checks E-1 .. E-5 of docs/specs/GNN3_CLEAN.md: the clean package must reproduce the published GNN3
numbers. Each command writes its part to gnn3/results/E_parts/<check>.json; ``merge`` writes E_equivalence.json.

    python gnn3/tests/equivalence.py e1      data arrays / lists / standardisation vs ch3 Ch3Data       (CPU)
    python gnn3/tests/equivalence.py e2      25 original model files -> fold test pairs vs KM_predictions (RTX 3090)
    python gnn3/tests/equivalence.py e3      fold 0 retrained from scratch vs D__lf3__S__GNN__f0          (RTX 3090)
    python gnn3/tests/equivalence.py e4      metrics vs EM_metrics / EM_rank / EA_cause / EM_identity     (CPU)
    python gnn3/tests/equivalence.py e5-select   choose the 10 E-5 reactions (2 per dipole, seed 20261111)  (CPU)
    python gnn3/tests/equivalence.py e5      input wrapper outputs of the 10 reactions vs $CH3/lf            (CPU)
                                             (the two arms are built by scripts/prepare_inputs.py first)
    python gnn3/tests/equivalence.py merge   -> gnn3/results/E_equivalence.json (all_pass)

Exit 0 = pass, 3 = fail (a STOP of the clean-up).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]               # gnn3/
REPO = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
from gnn3 import constants as K  # noqa: E402

PARTS = PACKAGE_ROOT / "results" / "E_parts"
E3_OUT = K.SCRATCH / "e3"
TOLERANCE = dict(e2_yhat=1e-6, e3_val=1e-6, e3_yhat=1e-4, e5=1e-8)


def array_sha256(values) -> str:
    values = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256(f"{values.dtype.str}|{values.shape}|".encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def write_part(name: str, record: dict) -> int:
    PARTS.mkdir(parents=True, exist_ok=True)
    record = dict(record, check=name, written=time.strftime("%Y-%m-%d %H:%M:%S"),
                  slurm_job=os.environ.get("SLURM_JOB_ID"))
    temporary = PARTS / f".{name}.json.tmp"
    temporary.write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
    temporary.replace(PARTS / f"{name}.json")
    print(json.dumps({k: record[k] for k in ("check", "ok") if k in record}, indent=1))
    return 0 if record.get("ok") else 3


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def original_data():
    """ch3 Ch3Data (= the data object of the published GNN3 units); binds the lf input paths like the original jobs."""
    sys.path.insert(0, str(REPO / "analysis" / "ch3"))
    import data_ch3 as DC3                                       # noqa: E402  (read-only use of the original code)
    DC3.use_targets(list(K.TARGETS))
    return DC3, DC3.get_data("ch3")


# --------------------------------------------------------------------------- E-1
def check_e1() -> int:
    from gnn3.data import PairDataset, Standardizer
    DC3, old = original_data()
    new = PairDataset()
    pairs = {
        "element_onehot": (old.onehot, new.reactions.element_onehot), "binary_flags": (old.bin, new.reactions.binary_flags),
        "numeric_features": (old.num, new.reactions.numeric_features),
        "positions_angstrom": (old.pos, new.reactions.positions_angstrom),
        "atom_mask": (old.mask, new.reactions.atom_mask), "edge_types": (old.etype, new.reactions.edge_types),
        "fragment_ids": (old.frag, new.reactions.fragment_ids), "n_atoms": (old.n, new.reactions.n_atoms),
        "global_features_182": (old.glob, new.reactions.global_features), "reaction_ids": (old.rids, new.reactions.reaction_ids),
        "targets_y4": (old.y, new.targets_kcal), "labels_y10": (old.y10, new.labels_kcal),
        "rows_a": (old.ra, new.rows_a), "rows_b": (old.rb, new.rows_b), "usable_pair": (old.usable_pair, new.usable_pair),
        "usable_reaction": (old.usable_rxn, new.usable_reaction), "pair_list": (np.array(old.pair_list), np.array(new.pair_list)),
    }
    checks = {name: array_sha256(a) == array_sha256(b) for name, (a, b) in pairs.items()}
    folds = {}
    for k in list(range(K.N_FOLDS)) + ["all"]:
        a = old.split(k)
        b = new.fold_split(k) if k != "all" else new.all_training_pairs()
        for name in K.FOLD_LISTS:
            folds[f"{k}/{name}"] = array_sha256(np.asarray(a[name], dtype=int)) == array_sha256(np.asarray(b[name], dtype=int))
        folds[f"{k}/pairs_sha256"] = a["pairs_sha"] == b.pairs_sha256
    standardisation = {}
    for k in range(K.N_FOLDS):
        train = new.fold_split(k).train
        s_old, s_new = DC3.FT.Std(old, train), Standardizer.from_training_pairs(new, train)
        for old_name, new_name in (("mu", "numeric_mean"), ("sd", "numeric_std"), ("gmu", "global_mean"),
                                   ("gsd", "global_std"), ("sy", "target_scale"), ("rxn_rows", "reaction_rows")):
            standardisation[f"{k}/{new_name}"] = array_sha256(getattr(s_old, old_name)) == array_sha256(getattr(s_new, new_name))
    import torch
    train0 = new.fold_split(0).train
    tensors_old = DC3.FT.Std(old, train0).tensors(old, device=torch.device("cpu"))
    tensors_new = Standardizer.from_training_pairs(new, train0).tensors(new.reactions, device=torch.device("cpu"))
    names = dict(x="atom_features", glob="global_features", pos="positions_angstrom", mask="atom_mask", etype="edge_types",
                 frag="fragment_ids", n="n_atoms")
    tensor_checks = {f"fold0/{new_name}": array_sha256(tensors_old[old_name].numpy())
                     == array_sha256(tensors_new[new_name].numpy()) for old_name, new_name in names.items()}
    ok = all(checks.values()) and all(folds.values()) and all(standardisation.values()) and all(tensor_checks.values())
    return write_part("E-1", dict(ok=ok, rule="sha256 of dtype, shape and bytes equal (new package vs ch3 Ch3Data)",
                                  arrays=checks, folds=folds, standardisation=standardisation, tensors=tensor_checks,
                                  n_pairs_usable=int(new.usable_pair.sum()), n_pair_list=len(new.pair_list)))


# --------------------------------------------------------------------------- E-2
def check_e2() -> int:
    import torch
    from gnn3.data import PairDataset
    from gnn3.predict import load_network, predict_with_networks, summarize_networks
    dataset = PairDataset()
    km = pd.read_parquet(K.ORIGINAL_RESULTS / "KM_predictions.parquet")
    km = km[(km.rung == "lf3") & (km.model == "GNN")]
    per_fold, worst, worst_sd = {}, 0.0, 0.0
    for k in range(K.N_FOLDS):
        test = dataset.fold_split(k).test
        files = [K.ORIGINAL_UNITS / f"D__lf3__S__GNN__f{k}__s{j}.pt" for j in range(5)]
        networks = [load_network(p) for p in files]
        mean, spread = summarize_networks(predict_with_networks(networks, dataset.reactions, dataset.rows_a[test],
                                                                dataset.rows_b[test]))
        ids = dataset.pair_ids(test)
        ref = km[km.fold == k].pivot(index="pair_id", columns="target", values="yhat").loc[ids, list(K.TARGETS)].to_numpy()
        ref_sd = km[km.fold == k].pivot(index="pair_id", columns="target", values="yhat_seed_sd").loc[ids, list(K.TARGETS)]
        diff, diff_sd = float(np.abs(mean - ref).max()), float(np.abs(spread - ref_sd.to_numpy()).max())
        per_fold[str(k)] = dict(n_test_pairs=len(ids), max_abs_diff_yhat=diff, max_abs_diff_seed_sd=diff_sd,
                                bit_identical=bool(np.array_equal(mean, ref)))
        worst, worst_sd = max(worst, diff), max(worst_sd, diff_sd)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return write_part("E-2", dict(ok=bool(worst <= TOLERANCE["e2_yhat"] and "3090" in gpu), tolerance=TOLERANCE["e2_yhat"],
                                  max_abs_diff_yhat=worst, max_abs_diff_seed_sd=worst_sd, folds=per_fold, gpu=gpu))


# --------------------------------------------------------------------------- E-3
def check_e3() -> int:
    import torch
    from gnn3.data import PairDataset
    from gnn3.training import run_cross_validation_unit
    dataset = PairDataset()
    record = run_cross_validation_unit(dataset, 0, E3_OUT)
    original = json.loads((K.ORIGINAL_UNITS / "D__lf3__S__GNN__f0.json").read_text())
    new_pred = pd.read_parquet(E3_OUT / "gnn3_fold0.parquet")
    old_pred = pd.read_parquet(K.ORIGINAL_UNITS / "D__lf3__S__GNN__f0.parquet")
    merged = old_pred.merge(new_pred, on=["pair_id", "target"], suffixes=("_old", "_new"), how="outer", indicator=True)
    same_pairs = bool((merged._merge == "both").all())
    tune = []
    for a, b in zip(original["tune"], record["tune"]):
        tune.append(dict(cell=b["cell"], same_cell=a["cell"] == b["cell"], best_epoch_old=a["best_epoch"],
                         best_epoch_new=b["best_epoch"], val_abs_diff=abs(a["val"] - b["val"]),
                         val_kcal_abs_diff=max(abs(a["val_kcal_mae"][t] - b["val_kcal_mae"][t]) for t in K.TARGETS)))
    yhat_diff = float(np.abs(merged.yhat_old - merged.yhat_new).max()) if same_pairs else float("inf")
    checks = dict(same_cell=original["chosen"]["cell"] == record["chosen"]["cell"],
                  same_best_epoch=int(original["chosen"]["best_epoch"]) == int(record["chosen"]["best_epoch"]),
                  tuning_same_cells_and_epochs=all(t["same_cell"] and t["best_epoch_old"] == t["best_epoch_new"] for t in tune),
                  validation_within_1e6=max(max(t["val_abs_diff"], t["val_kcal_abs_diff"]) for t in tune) <= TOLERANCE["e3_val"],
                  test_yhat_within_1e4=yhat_diff <= TOLERANCE["e3_yhat"], same_test_pairs=same_pairs)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return write_part("E-3", dict(ok=bool(all(checks.values()) and "3090" in gpu), checks=checks, tune=tune,
                                  chosen_new=record["chosen"], chosen_original=original["chosen"],
                                  max_abs_diff_test_yhat=yhat_diff,
                                  bit_identical_test_yhat=bool(same_pairs and (merged.yhat_old == merged.yhat_new).all()),
                                  wall_seconds=record["wall_seconds"], gpu=gpu, tolerance=TOLERANCE))


# --------------------------------------------------------------------------- E-4
def check_e4() -> int:
    from gnn3 import metrics as M
    truth, attributes = M.channel_truth()
    sets = M.pair_sets()
    km = pd.read_parquet(K.ORIGINAL_RESULTS / "KM_predictions.parquet")
    gnn3 = km[(km.rung == "lf3") & (km.model == "GNN")][["pair_id", "fold", "target", "yhat", "yhat_seed_sd"]].copy()
    gnn3["pair_id"] = gnn3.pair_id.astype(str)
    long = M.assemble({"GNN3": gnn3}, truth, attributes, sets)
    ids = sorted(long.pair_id.unique())
    out = {}

    def compare(name, mine: pd.DataFrame, theirs: pd.DataFrame, keys, columns):
        merged = theirs.merge(mine, on=keys, suffixes=("_ref", "_new"), how="left")
        checks = {}
        for column in columns:
            a, b = merged[f"{column}_ref"].to_numpy(float), merged[f"{column}_new"].to_numpy(float)
            both_nan = np.isnan(a) & np.isnan(b)
            checks[column] = bool(((np.round(a, 4) == np.round(b, 4)) | both_nan).all())
        out[name] = dict(n_rows=int(len(merged)), columns=checks, ok=bool(len(merged) and all(checks.values())))

    em = pd.read_csv(K.ORIGINAL_RESULTS / "EM_metrics.csv")
    em = em[em.model == "GNN3"].drop(columns=["model"])
    compare("EM_metrics (every GNN3 row)", M.error_table(long, ids, ["GNN3"]).drop(columns=["model"]), em,
            ["eval_set", "target", "pos"], ["n", "mae", "rmse", "r", "sign_acc", "n_sign", "mae_ci_lo", "mae_ci_hi"])
    er = pd.read_csv(K.ORIGINAL_RESULTS / "EM_rank.csv")
    er = er[er.model == "GNN3"].drop(columns=["model"])
    rank_columns = [c for c in er.columns if c not in ("scope", "rank_set")]
    compare("EM_rank (GNN3 para + all)", M.ranking_table(long, ids, ["GNN3"]).drop(columns=["model"]), er,
            ["scope", "rank_set"], rank_columns)
    ec = pd.read_csv(REPO / "analysis/abs3/results/EA_cause.csv")
    ec = ec[ec.model == "GNN3"].drop(columns=["model", "true_dist", "baseline_class"])
    mine = M.cause_table([M.cause_agreement(long, ids, "GNN3")]).drop(columns=["model", "true_dist", "baseline_class"])
    compare("EA_cause (GNN3 rows)", mine, ec, ["level", "measure", "cause_set", "sign_mode"],
            ["n", "agreement", "baseline", "r", "mae"])
    ei = pd.read_csv(K.ORIGINAL_RESULTS / "EM_identity.csv")
    ei = ei[ei.model == "GNN3"]
    compare("EM_identity (GNN3)", pd.DataFrame([M.identity_residual(long, ids, "GNN3")]), ei, ["model"],
            ["yhat_mean_abs", "yhat_p95_abs", "yhat_max_abs", "y_max_abs", "n"])
    em_ad = json.loads((K.ORIGINAL_RESULTS / "EM_ad.json").read_text())["thresholds"]["GNN3"]
    mine_ad = M.applicability_thresholds(gnn3)
    out["EM_ad (GNN3 thresholds)"] = dict(ok=all(round(mine_ad[t], 4) == round(em_ad[t], 4) for t in K.TARGETS),
                                          new=mine_ad, original=em_ad)
    return write_part("E-4", dict(ok=all(v["ok"] for v in out.values()), rule="4 decimals", n_pairs=len(ids), tables=out))


# --------------------------------------------------------------------------- E-5
E5_ROOT = K.SCRATCH / "e5"
E5_SEED, E5_PER_DIPOLE = 20261111, 2
E5_ARMS = {"reuse": "finalize folders copied from $CH3/lf (no ORCA rerun); metadata, features, tokens recomputed",
           "rerun": "everything from the existing g2 TS folders recomputed, including the ORCA frequency job"}
#: The 182 features that come from the ORCA Hessian (ext block B6, read from work/ts.hess): only these may differ
#: when the frequency job is rerun.
HESSIAN_FEATURES = ("nu_imag", "mode_share", "mode_async")
TOKEN_ARRAYS = ("rxn_id", "Z", "bin", "num", "pos", "edges", "etype", "glob", "n_bonds_rel", "token_version")


def check_e5_select() -> int:
    """Seeded choice of the 10 E-5 reactions (2 per dipole) -> $GNN3/e5/list.csv (job_id) + selection.csv."""
    from gnn3.inputs.pipeline import select_reactions
    table = pd.read_parquet(K.LF_INPUTS / "X182_lf.parquet")
    finite = np.isfinite(table.drop(columns=["rxn_id"]).to_numpy(float)).all(1)
    rows = select_reactions(E5_PER_DIPOLE, E5_SEED, K.TOKEN_REPORTS / "T_manifest.csv",
                            set(table.rxn_id[finite].astype(int)))
    E5_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(job_id=[r["job_id"] for r in rows])).to_csv(E5_ROOT / "list.csv", index=False)
    pd.DataFrame(rows)[["job_id", "rxn_id", "dipole", "status"]].to_csv(E5_ROOT / "selection.csv", index=False)
    print(pd.DataFrame(rows)[["job_id", "rxn_id", "dipole"]].to_string(index=False))
    return 0 if len(rows) == 10 else 3


def _compare_arm(root: Path, rxn_ids: list[int]) -> dict:
    """Max |difference| of every token array and every 182 feature of ``root`` against the published $CH3/lf."""
    out = dict(root=str(root), steps={})
    for step in ("finalize", "metadata", "features", "tokens"):
        path = root / "results" / "gnn3_steps" / f"{step}.json"
        out["steps"][step] = json.loads(path.read_text()) if path.is_file() else None
    if not all(s is not None and s.get("rc") == 0 for s in out["steps"].values()):
        return dict(out, complete=False)
    new_table = pd.read_parquet(root / "X182_lf.parquet").set_index("rxn_id")
    old_table = pd.read_parquet(K.LF_INPUTS / "X182_lf.parquet").set_index("rxn_id")
    columns = list(old_table.columns)
    missing = [r for r in rxn_ids if r not in new_table.index]
    common = [r for r in rxn_ids if r in new_table.index]
    new_values = new_table.loc[common, columns].to_numpy(float)
    old_values = old_table.loc[common, columns].to_numpy(float)
    nan_same = bool((np.isnan(new_values) == np.isnan(old_values)).all())
    difference = np.where(np.isnan(new_values) & np.isnan(old_values), 0.0, np.abs(new_values - old_values))
    x182_diff = dict(zip(columns, np.nan_to_num(difference, nan=np.inf).max(0).tolist())) if common else {}
    token_diff, token_missing, exact_equal, token_nan_same, per_reaction = {}, [], True, True, {}
    for rid in rxn_ids:
        new_path, old_path = root / "fu2_lf" / "tokens" / f"{rid}.npz", K.LF_INPUTS / "fu2_lf" / "tokens" / f"{rid}.npz"
        if not new_path.is_file():
            token_missing.append(rid)
            continue
        new, old = np.load(new_path), np.load(old_path)
        record = {}
        for key in TOKEN_ARRAYS:
            a, b = new[key], old[key]
            if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
                record[key] = f"shape / type differ: {a.shape} {a.dtype} vs {b.shape} {b.dtype}"
                exact_equal = False
            elif a.dtype.kind == "f":
                same_nan = bool((np.isnan(a) == np.isnan(b)).all())
                token_nan_same = token_nan_same and same_nan
                d = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
                d = np.nan_to_num(d, nan=np.inf)
                if key == "glob":                              # the token copy of the 182 features, per column
                    for c, v in zip(columns, d.tolist()):
                        token_diff[f"glob:{c}"] = max(token_diff.get(f"glob:{c}", 0.0), v)
                    record[key] = float(d.max()) if d.size else 0.0
                else:
                    token_diff[key] = max(token_diff.get(key, 0.0), float(d.max()) if d.size else 0.0)
                    record[key] = token_diff[key]
            else:
                equal = bool(np.array_equal(a, b))
                exact_equal = exact_equal and equal
                record[key] = "equal" if equal else "DIFFERENT"
        per_reaction[str(rid)] = record
    return dict(out, complete=True, n_reactions=len(rxn_ids), x182_same_columns=list(new_table.columns) == columns,
                x182_missing=missing, x182_nan_same=nan_same, x182_max_abs_diff=x182_diff,
                token_missing=token_missing, token_exact_arrays_equal=exact_equal, token_nan_same=token_nan_same,
                token_max_abs_diff=token_diff, per_reaction=per_reaction)


def _over(values: dict, tolerance: float, allowed=()) -> list[str]:
    return sorted(k for k, v in values.items() if v > tolerance and k.split(":")[-1] not in allowed)


def check_e5() -> int:
    """E-5: both arms against $CH3/lf. Rule (DECISIONS, written before the run): arm 'reuse' <= 1e-8 on every value;
    arm 'rerun' <= 1e-8 on every value except the 3 Hessian features (reported); any missing output = fail."""
    selection = pd.read_csv(E5_ROOT / "selection.csv", dtype={"job_id": str})
    rxn_ids = selection.rxn_id.astype(int).tolist()
    arms, ok = {}, True
    for arm in E5_ARMS:
        result = _compare_arm(E5_ROOT / arm, rxn_ids)
        arm_ok = False
        if result.get("complete"):
            allowed = HESSIAN_FEATURES if arm == "rerun" else ()
            over = _over(result["x182_max_abs_diff"], TOLERANCE["e5"], allowed) + \
                _over(result["token_max_abs_diff"], TOLERANCE["e5"], allowed)
            arm_ok = bool(not over and result["x182_same_columns"] and not result["x182_missing"]
                          and not result["token_missing"] and result["token_exact_arrays_equal"]
                          and result["x182_nan_same"] and result["token_nan_same"])
            values = {**result["x182_max_abs_diff"], **result["token_max_abs_diff"]}
            result.update(over_tolerance=over,
                          hessian_feature_max_abs_diff={c: max(result["x182_max_abs_diff"].get(c, 0.0),
                                                               result["token_max_abs_diff"].get(f"glob:{c}", 0.0))
                                                        for c in HESSIAN_FEATURES},
                          max_abs_diff_all=max(values.values()),
                          max_abs_diff_excluding_hessian=max(v for k, v in values.items()
                                                             if k.split(":")[-1] not in HESSIAN_FEATURES))
        result.update(ok=arm_ok, description=E5_ARMS[arm])
        arms[arm] = result
        ok = ok and arm_ok
    return write_part("E-5", dict(ok=ok, tolerance=TOLERANCE["e5"], seed=E5_SEED, per_dipole=E5_PER_DIPOLE,
                                  selection=selection.to_dict("records"), hessian_features=list(HESSIAN_FEATURES),
                                  rule="reuse arm <= 1e-8 on every value; rerun arm <= 1e-8 except the 3 Hessian "
                                       "features (ORCA frequency rerun; reported)", arms=arms,
                                  not_rerun="the g2 TS search (autodE conformer search is random)"))


# --------------------------------------------------------------------------- merge
def merge() -> int:
    parts = {}
    for name in ("E-1", "E-2", "E-3", "E-4", "E-5", "E-6"):
        path = PARTS / f"{name}.json"
        parts[name] = json.loads(path.read_text()) if path.is_file() else dict(ok=False, missing=True)
    record = dict(spec="docs/specs/GNN3_CLEAN.md 5", written=time.strftime("%Y-%m-%d %H:%M:%S"),
                  all_pass=all(p.get("ok") for p in parts.values()),
                  summary={k: bool(v.get("ok")) for k, v in parts.items()}, checks=parts)
    path = PACKAGE_ROOT / "results" / "E_equivalence.json"
    path.write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
    print(json.dumps(record["summary"], indent=1))
    return 0 if record["all_pass"] else 3


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    handlers = {"e1": check_e1, "e2": check_e2, "e3": check_e3, "e4": check_e4, "e5-select": check_e5_select,
                "e5": check_e5, "merge": merge}
    if command not in handlers:
        print(__doc__)
        sys.exit(2)
    sys.exit(handlers[command]())
