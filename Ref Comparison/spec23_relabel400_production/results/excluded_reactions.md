# Excluded reactions from spec23 final label parquet

## dipolar_003313

**Reason**: source charge/multiplicity data error.

The ORCA input generated from `labels/orca/orca_eda_charges_v9.parquet`
had charge=0, multiplicity=1 for both fragments. But the actual electron
count is odd (31 in one fragment, 39 in the other), which is
mathematically incompatible with a closed-shell singlet:

```
Error: multiplicity (1) is odd and number of electrons (31) is odd -> impossible
```

All three ORCA jobs (eda, fragA_opt, fragB_opt) rejected the input for
this reason. The reaction is likely a radical/open-shell case
(multiplicity ≥ 2) or an ion (charge ≠ 0), but the exact assignment is
unclear from available source data.

## Impact
- Final cohort: **399** instead of 400
  - locked_778: 192 (unchanged)
  - spec16: 207 (was 208, dipolar_003313 excluded)
- All downstream stages must use 399 as the reference count.
- Spec21's D2/D3 diagnostics were run on 400, but conclusions
  (halves geometry-homogeneous, no scaffold enrichment) hold on 399.
