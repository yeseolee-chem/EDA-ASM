#!/usr/bin/env python3
"""Step 4: build MMP pairs.

For each reaction × each cut candidate:
  - Remove the "sub" side (smaller half by heavy count, break ties by lex)
  - Replace with [*], canonicalize the resulting reactant
  - Include the OTHER reactant molecule in the key (join with '.', sorted)
  - Key = masked_reactant_smiles + '|' + solvent + '|' + f"{temp:.2f}"

Bucket key -> list of (sub_smi, rxn_id). Within each bucket, form all pairs where
sub_A != sub_B, rxn_A != rxn_B, order (sub_A_canon < sub_B_canon by lex).

Dedup on (rxn_id_A, rxn_id_B, key).
"""
from pathlib import Path
from collections import defaultdict
import pandas as pd
from rdkit import Chem

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/mmp_gate_a")
OUT = BASE / "artifacts"


def canon(smi: str) -> str:
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return smi
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    for a in m.GetAtoms():
        if a.GetSymbol() == "*":
            a.SetAtomMapNum(0)
            a.SetIsotope(0)
    return Chem.MolToSmiles(m, canonical=True)


def sort_join(*smis):
    return ".".join(sorted(canon(s) for s in smis if s))


def main():
    cohort = pd.read_pickle(OUT / "cohort_join.pkl")
    frags = pd.read_pickle(OUT / "fragments.pkl")
    print(f"Cohort: {len(cohort)}  Fragments: {len(frags)}")

    cohort_ix = {int(r["reaction_number"]): r for _, r in cohort.iterrows()}

    # Cache reactant molecule SMILES per reaction (2 mols each)
    react_smi = {}
    for rxn, r in cohort_ix.items():
        parts = r["rxn_smiles"].split(">>")[0].split(".")
        react_smi[rxn] = [canon(_strip_map(p)) for p in parts]

    index = defaultdict(list)
    n_cuts_used = 0
    for _, row in frags.iterrows():
        rxn = int(row["rxn_id"])
        mol_idx = int(row["mol_idx"])
        a_smi, b_smi = row["frag_a_smiles"], row["frag_b_smiles"]
        ah, bh = row["frag_a_heavy"], row["frag_b_heavy"]
        # smaller side = sub, larger side = mask
        if ah < bh:
            sub_smi, mask_smi = a_smi, b_smi
        elif bh < ah:
            sub_smi, mask_smi = b_smi, a_smi
        else:
            # tie: pick lex-smaller as sub for determinism
            if canon(a_smi) <= canon(b_smi):
                sub_smi, mask_smi = a_smi, b_smi
            else:
                sub_smi, mask_smi = b_smi, a_smi
        # Build full masked reactant: masked mol_idx + other mol_idx unchanged
        parts = react_smi[rxn]
        other = [p for i, p in enumerate(parts) if i != mol_idx]
        masked_full = sort_join(mask_smi, *other)
        r = cohort_ix[rxn]
        key = f"{masked_full}|{r['solvent']}|{float(r['temp']):.2f}"
        index[key].append((canon(sub_smi), rxn))
        n_cuts_used += 1

    print(f"Bucketed cuts: {n_cuts_used}  Unique keys: {len(index)}")

    pairs = []
    seen = set()
    for key, entries in index.items():
        # Group by rxn to reduce redundancy: keep unique (sub, rxn) tuples
        uniq = list(set(entries))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                sub_a, rxn_a = uniq[i]
                sub_b, rxn_b = uniq[j]
                if sub_a == sub_b or rxn_a == rxn_b:
                    continue
                # deterministic ordering by sub_smi
                if sub_a > sub_b:
                    sub_a, sub_b = sub_b, sub_a
                    rxn_a, rxn_b = rxn_b, rxn_a
                dedup = (rxn_a, rxn_b, key)
                if dedup in seen:
                    continue
                seen.add(dedup)
                pairs.append({
                    "rxn_id_A": rxn_a, "rxn_id_B": rxn_b,
                    "sub_A": sub_a, "sub_B": sub_b, "key": key,
                    "G_act_A": cohort_ix[rxn_a]["G_act"],
                    "G_act_B": cohort_ix[rxn_b]["G_act"],
                    "delta_G_act": cohort_ix[rxn_b]["G_act"] - cohort_ix[rxn_a]["G_act"],
                })

    df = pd.DataFrame(pairs)
    df.to_pickle(OUT / "mmp_pairs.pkl")
    print(f"MMP pairs (unique): {len(df)}")
    if len(df) == 0:
        print("=== GATE-3 FAIL: 0 pairs ===")
    else:
        print(f"=== GATE-3 report: {len(df)} pairs, unique rxns involved: "
              f"{len(set(df['rxn_id_A'])|set(df['rxn_id_B']))} ===")


def _strip_map(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m, canonical=True)


if __name__ == "__main__":
    main()
