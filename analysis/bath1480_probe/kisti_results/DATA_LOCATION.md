# Data folder note

`data/` is a **symbolic link** to the actual location in scratch:
```
/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data/
```

This avoids duplicating 664 MB (3,504 reactions × 3 files) into home1
(which is gitignored anyway). VS Code / any file browser can follow the
symlink transparently.

If the scratch location is ever cleaned, this symlink will break — the
original tar (`tt_eda_full_20260808.tar.gz`, 146 MB) should be kept
somewhere as backup, or re-extracted from wherever the KISTI transfer
originated.
