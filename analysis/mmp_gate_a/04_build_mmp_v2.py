#!/usr/bin/env python3
"""Step 4 (v2 patch): MMP construction with 3 filters (§3 of rev SPEC).

Filter 1: exactly one fragment contains core atoms (the other = 'sub')
Filter 2: sub heavy count <= SUB_MAX_HEAVY (=8)
Filter 3: core_signature explicitly in key
"""
from pathlib import Path
from collections import defaultdict
import pandas as pd
from rdkit import Chem

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"
SUB_MAX_HEAVY = 8
EXPECTED_PAIRS = 2207
EXPECTED_TOL = 0.05  # ±5%


def canon(smi):
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return smi
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
        if a.GetSymbol() == "*":
            a.SetIsotope(0)
    return Chem.MolToSmiles(m, canonical=True)


def strip_and_canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True)


def sort_join(*smis):
    return ".".join(sorted(canon(s) for s in smis if s))


def main():
    cohort = pd.read_pickle(OUT / "cohort_join.pkl")
    frags = pd.read_pickle(OUT / "fragments_v2.pkl")
    print(f"Cohort: {len(cohort)}  Fragments(v2): {len(frags)}")

    cohort_ix = {int(r["reaction_number"]): r for _, r in cohort.iterrows()}
    react_smi = {}
    for rxn, r in cohort_ix.items():
        parts = r["rxn_smiles"].split(">>")[0].split(".")
        react_smi[rxn] = [strip_and_canon(p) for p in parts]

    # === Filter 1 + 2 ===
    n_both_core = n_neither = n_sub_too_big = n_ok = 0
    index = defaultdict(list)
    for _, row in frags.iterrows():
        a_core, b_core = bool(row["frag_a_has_core"]), bool(row["frag_b_has_core"])
        if a_core and b_core:
            n_both_core += 1
            continue
        if not a_core and not b_core:
            # should not happen — core is always in one of the two fragments
            n_neither += 1
            continue
        # Filter 1: sub is the fragment WITHOUT core
        if a_core:
            sub_smi, mask_smi = row["frag_b_smiles"], row["frag_a_smiles"]
            sub_h = int(row["frag_b_heavy"])
        else:
            sub_smi, mask_smi = row["frag_a_smiles"], row["frag_b_smiles"]
            sub_h = int(row["frag_a_heavy"])

        # Filter 2: sub size
        if sub_h > SUB_MAX_HEAVY:
            n_sub_too_big += 1
            continue

        rxn = int(row["rxn_id"])
        mol_idx = int(row["mol_idx"])
        parts = react_smi[rxn]
        other = [p for i, p in enumerate(parts) if i != mol_idx]
        masked_full = sort_join(mask_smi, *other)
        r = cohort_ix[rxn]
        csig = row["core_signature"]
        # Filter 3: core_signature explicitly in key
        key = f"{masked_full}|{csig}|{r['solvent']}|{float(r['temp']):.2f}"
        index[key].append((canon(sub_smi), sub_h, rxn))
        n_ok += 1

    print(f"Filter stats: kept={n_ok}  both_core={n_both_core}  neither={n_neither}  sub_too_big={n_sub_too_big}")
    print(f"Unique bucket keys: {len(index)}")

    pairs = []
    seen = set()
    for key, entries in index.items():
        uniq = list(set(entries))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                sub_a, h_a, rxn_a = uniq[i]
                sub_b, h_b, rxn_b = uniq[j]
                if sub_a == sub_b or rxn_a == rxn_b:
                    continue
                if sub_a > sub_b:
                    sub_a, sub_b = sub_b, sub_a
                    h_a, h_b = h_b, h_a
                    rxn_a, rxn_b = rxn_b, rxn_a
                dd = (rxn_a, rxn_b, key)
                if dd in seen:
                    continue
                seen.add(dd)
                # Extract core_signature back out of key (between last 3 '|')
                # key = masked|core_sig|solvent|temp
                parts_key = key.split("|")
                csig = parts_key[-3] if len(parts_key) >= 4 else ""
                pairs.append({
                    "rxn_id_A": rxn_a, "rxn_id_B": rxn_b,
                    "sub_A": sub_a, "sub_B": sub_b,
                    "sub_n_heavy_A": h_a, "sub_n_heavy_B": h_b,
                    "core_signature": csig,
                    "key": key,
                    "G_act_A": cohort_ix[rxn_a]["G_act"],
                    "G_act_B": cohort_ix[rxn_b]["G_act"],
                    "delta_G_act": cohort_ix[rxn_b]["G_act"] - cohort_ix[rxn_a]["G_act"],
                })

    df = pd.DataFrame(pairs)
    df.to_pickle(OUT / "mmp_pairs_v2.pkl")
    n_rxns = len(set(df["rxn_id_A"]) | set(df["rxn_id_B"])) if len(df) else 0
    print(f"MMP pairs (v2, unique): {len(df)}  reactions involved: {n_rxns}")

    lo = int(EXPECTED_PAIRS * (1 - EXPECTED_TOL))
    hi = int(EXPECTED_PAIRS * (1 + EXPECTED_TOL))
    if lo <= len(df) <= hi:
        stat = "PASS"
    else:
        stat = "FAIL"
    with open(OUT / "GATE3rev_STATUS.txt", "w") as f:
        f.write(f"{stat} n_pairs={len(df)} expected={EXPECTED_PAIRS}±5% [{lo},{hi}]\n")
    print(f"=== GATE-3-rev: {stat} n_pairs={len(df)} (expected {EXPECTED_PAIRS}±5%) ===")


if __name__ == "__main__":
    main()
