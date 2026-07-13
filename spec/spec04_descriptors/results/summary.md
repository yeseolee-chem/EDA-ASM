# SPEC_04 (XGB-based descriptor contribution) - summary

- Cohort: 783 rxns (m3 v9 783), D = 28
- Method: per-channel XGB (fixed HP matching SPEC_03/SPEC_06),
  forward-selection with 3-fold inner CV NMAE, gain-based importance heatmap.
- Collinearity: cond(X.T@X) = 2.29e+16, rank = 28/28

## Top-5 forward-selection order per channel

| channel | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| strain | d25 | d7 | d27 | d21 | d19 |
| Pauli | d27 | d10 | d15 | d13 | d7 |
| elst | d24 | d28 | d27 | d7 | d13 |
| oi | d24 | d27 | d7 | d10 | d21 |
| disp | d10 | d3 | d27 | d9 | d13 |
| barrier | d10 | d25 | d7 | d24 | d27 |

## Union of top-8 across channels (20 descriptors)
- ['d3', 'd6', 'd7', 'd9', 'd10', 'd13', 'd14', 'd15', 'd16', 'd17', 'd19', 'd20', 'd21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28']

## VIF flags (severe = VIF>10, moderate = VIF>5)
See vif.csv.

## Reduced set (union of per-channel elbows)
- Core (14): ['d10', 'd13', 'd15', 'd16', 'd19', 'd21', 'd24', 'd25', 'd26', 'd27', 'd28', 'd3', 'd7', 'd9']
- See reduced_set_delta.csv for NMAE (full) vs NMAE (reduced) per channel.