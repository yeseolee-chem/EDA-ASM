# ADF_800 — Initial Seed Run Archive

Snapshot created 2026-06-01 (job halted by user). Self-contained archive of
the BLYP-D3(BJ)/TZ2P ADF EDA-ASM run on the 794-reaction initial seed.

## Status at archive time

- **Total queued**: 794 reactions (planned 800; 6 dropped at fragmentation)
- **status.json written**: 213 (209 converged + 4 failed)
- **Validly parsed → asr_labels.parquet**: 135 reactions
  - 134 dipolar cycloaddition
  - 1 qmrxn20_e2
- **Remaining**: 78 status.json files are May-29 relics from a pre-rebuild
  PBE0 run; `.out` files were wiped during the May-30 disk-crisis cleanup,
  so the BLYP-D3 parser skips them. They are kept for provenance but
  contribute no labels.

### Per-batch breakdown

| batch | reactions | status | converged | notes |
|---|---|---|---|---|
| batch_000 | 100 | 100 | 99 | 1 failed |
| batch_001 | 100 | 100 | 97 | 3 failed |
| batch_002 | 100 | 13 | 13 | aborted mid-batch |
| batch_003–007 | 494 | 0 | 0 | input decks only |

## Layout

```
ADF_800/
├── MANIFEST.md                 (this file)
├── adf_outputs/                (MOVED from repo)
│   ├── batch_000…007/          per-reaction C1–C5 in/out, status.json
│   ├── batch_manifest.json     794-reaction inventory + provenance
│   ├── run_batch.sh            xargs-P4 runner
│   ├── submit_all.sh           master driver
│   └── parsed/
│       └── asr_labels.parquet  ← FINAL LABELS (135 rows × 5-channel ASR vector)
├── seed_selection/             (MOVED from data/selection/)
│   └── initial_seed_v1/        Kennard-Stone 3-tier stratified pick
│       ├── selected_reactions.csv
│       ├── selected_reaction_ids.json
│       ├── pool_after_conformer_collapse.parquet
│       ├── morgan_fingerprints.npy
│       ├── stratification_diagnostics.png
│       └── manifest.json
├── logs/                       (MOVED from logs/production/)
│   ├── main.log                first production attempt
│   ├── main2.log               final production driver log
│   └── .pid                    driver PID record (3221301)
├── configs/                    (COPIED — originals remain in repo)
│   ├── adf_computation.yaml    BLYP-D3(BJ)/TZ2P Normal-quality protocol
│   ├── fragment_partitioning.yaml  SMARTS patterns per family
│   └── seed_selection.yaml     3-tier stratification weights + seed
├── scripts/                    (COPIED — originals remain in repo)
│   ├── adf/
│   │   ├── generate_adf_inputs.py  C1–C5 deck builder
│   │   ├── write_status.py         per-reaction status.json writer
│   │   ├── check_adf_status.py     roll-up over batches
│   │   ├── extract_asr_labels.py   produces asr_labels.parquet
│   │   ├── aggregate_results.py
│   │   ├── parse_run.py
│   │   └── build_all_inputs.py
│   ├── define_fragments.py     SMARTS skeleton-query fragment splitter
│   └── select_initial_seed.py  reproducible 800-pick driver
└── src/                        (COPIED — originals remain in repo)
    └── eda_asm/
        ├── adf/                AdfRunner library (runner/parser/fragmentation)
        └── datasets/           native Stuyver + QMrxn20 loaders
```

## Reproducing

The configs + scripts + src/ in this archive are an exact snapshot of what
produced `adf_outputs/`. To re-run from scratch on the same seed:

```bash
# from a fresh repo with src/eda_asm package installed:
python scripts/select_initial_seed.py --config configs/seed_selection.yaml
python scripts/define_fragments.py    --config configs/fragment_partitioning.yaml
python scripts/adf/generate_adf_inputs.py --config configs/adf_computation.yaml --force
bash adf_outputs/submit_all.sh 4
python scripts/adf/extract_asr_labels.py
```

## Notes / caveats

- **6 reactions dropped at fragmentation** (800 → 794): connected-component
  cuts produced >2 fragments. See `seed_selection/initial_seed_v1/selection_log.txt`.
- **May-29 relics**: ~78 reaction dirs hold pre-rebuild status.json with
  `exit_code=0` but no `.out` files — they parse to nothing. Treat
  `asr_labels.parquet` as the authoritative label set.
- **License**: ADF (AMS 2026.103) at `/home1/yeseo1ee/ams2026.103`, license
  valid only on `gate1.hpc` (cni-podman0 MAC). Re-runs must use gate1.
- **Throughput**: ~2 reactions/h on 4-way xargs for the heavy dipolar
  cycloaddition systems; QMrxn20 (e2/sn2) expected ~10× faster.
