# 4-Phase Experiment (MAE + NMAE)

- Cohort: 926 rxns (5-fold × 5-seed CV, 25 evaluations per config)
- Phase 1 (Espley baseline): predicts 3 channels (d1, d2, interaction)
- Phase 2/3/4: predict 8 channels (d1, d2, elst, pauli, oi, disp, cpcm, cds)

## Summary (all configs × targets)

| phase                   | model   | target             |   mae_mean |   mae_std |   nmae_mean |   nmae_std |   r2_mean |
|:------------------------|:--------|:-------------------|-----------:|----------:|------------:|-----------:|----------:|
| phase1_espley           | krr     | d1_dft_espley      |     3.2124 |    0.1372 |      0.0871 |     0.0037 |    0.3843 |
| phase1_espley           | krr     | d2_dft_espley      |     3.1935 |    0.1547 |      0.1073 |     0.0052 |    0.3979 |
| phase1_espley           | krr     | interaction_espley |     2.3328 |    0.1338 |      0.0919 |     0.0053 |    0.5147 |
| phase1_espley           | ridge   | d1_dft_espley      |     3.4631 |    0.1897 |      0.0939 |     0.0051 |    0.3158 |
| phase1_espley           | ridge   | d2_dft_espley      |     3.3165 |    0.2206 |      0.1114 |     0.0074 |    0.3429 |
| phase1_espley           | ridge   | interaction_espley |     2.8457 |    0.1551 |      0.1121 |     0.0061 |    0.2817 |
| phase1_espley           | svr     | d1_dft_espley      |     3.1914 |    0.1523 |      0.0866 |     0.0041 |    0.3913 |
| phase1_espley           | svr     | d2_dft_espley      |     3.2445 |    0.2095 |      0.1090 |     0.0070 |    0.3635 |
| phase1_espley           | svr     | interaction_espley |     2.4053 |    0.1374 |      0.0947 |     0.0054 |    0.4237 |
| phase2_armB_tau1e-10    | krr     | cds_dft            |     0.4102 |    0.0552 |      0.0682 |     0.0092 |    0.4420 |
| phase2_armB_tau1e-10    | krr     | cpcm_dft           |     2.4054 |    0.2008 |      0.0528 |     0.0044 |    0.7339 |
| phase2_armB_tau1e-10    | krr     | d1_own             |     2.4776 |    0.0367 |      0.0655 |     0.0010 |    0.7903 |
| phase2_armB_tau1e-10    | krr     | d2_own             |     2.2456 |    0.1830 |      0.0468 |     0.0038 |    0.8116 |
| phase2_armB_tau1e-10    | krr     | disp_dft           |     0.0579 |    0.0016 |      0.0029 |     0.0001 |    0.9993 |
| phase2_armB_tau1e-10    | krr     | elst_dft           |     3.8420 |    0.4410 |      0.0378 |     0.0043 |    0.8092 |
| phase2_armB_tau1e-10    | krr     | oi_dft             |     3.3972 |    0.4598 |      0.0340 |     0.0046 |    0.8262 |
| phase2_armB_tau1e-10    | krr     | pauli_dft          |     5.7321 |    0.6439 |      0.0283 |     0.0032 |    0.8620 |
| phase2_armB_tau1e-10    | ridge   | cds_dft            |     0.4450 |    0.0457 |      0.0740 |     0.0076 |    0.3607 |
| phase2_armB_tau1e-10    | ridge   | cpcm_dft           |     2.4727 |    0.2096 |      0.0543 |     0.0046 |    0.7186 |
| phase2_armB_tau1e-10    | ridge   | d1_own             |     2.7406 |    0.1063 |      0.0725 |     0.0028 |    0.7421 |
| phase2_armB_tau1e-10    | ridge   | d2_own             |     2.4080 |    0.1689 |      0.0502 |     0.0035 |    0.7843 |
| phase2_armB_tau1e-10    | ridge   | disp_dft           |     0.0000 |    0.0000 |      0.0000 |     0.0000 |    1.0000 |
| phase2_armB_tau1e-10    | ridge   | elst_dft           |     4.0782 |    0.3635 |      0.0402 |     0.0036 |    0.8006 |
| phase2_armB_tau1e-10    | ridge   | oi_dft             |     3.7631 |    0.2980 |      0.0376 |     0.0030 |    0.7949 |
| phase2_armB_tau1e-10    | ridge   | pauli_dft          |     6.7297 |    0.7678 |      0.0333 |     0.0038 |    0.8189 |
| phase2_armB_tau1e-10    | svr     | cds_dft            |     0.4032 |    0.0497 |      0.0670 |     0.0083 |    0.4338 |
| phase2_armB_tau1e-10    | svr     | cpcm_dft           |     2.4513 |    0.1296 |      0.0538 |     0.0028 |    0.7174 |
| phase2_armB_tau1e-10    | svr     | d1_own             |     2.4013 |    0.1077 |      0.0635 |     0.0028 |    0.8275 |
| phase2_armB_tau1e-10    | svr     | d2_own             |     2.1870 |    0.1946 |      0.0456 |     0.0041 |    0.8056 |
| phase2_armB_tau1e-10    | svr     | disp_dft           |     0.0360 |    0.0046 |      0.0018 |     0.0002 |    0.9997 |
| phase2_armB_tau1e-10    | svr     | elst_dft           |     3.9129 |    0.3281 |      0.0385 |     0.0032 |    0.8066 |
| phase2_armB_tau1e-10    | svr     | oi_dft             |     3.2881 |    0.3424 |      0.0329 |     0.0034 |    0.8233 |
| phase2_armB_tau1e-10    | svr     | pauli_dft          |     6.2991 |    0.8549 |      0.0311 |     0.0042 |    0.8407 |
| phase3_armB_tau0.05     | krr     | cds_dft            |     0.4248 |    0.0313 |      0.0706 |     0.0052 |    0.4115 |
| phase3_armB_tau0.05     | krr     | cpcm_dft           |     2.5814 |    0.1938 |      0.0567 |     0.0043 |    0.6890 |
| phase3_armB_tau0.05     | krr     | d1_own             |     2.4819 |    0.0855 |      0.0656 |     0.0023 |    0.7864 |
| phase3_armB_tau0.05     | krr     | d2_own             |     2.2823 |    0.1620 |      0.0476 |     0.0034 |    0.8101 |
| phase3_armB_tau0.05     | krr     | disp_dft           |     0.0446 |    0.0026 |      0.0023 |     0.0001 |    0.9996 |
| phase3_armB_tau0.05     | krr     | elst_dft           |     4.6976 |    0.2863 |      0.0463 |     0.0028 |    0.7430 |
| phase3_armB_tau0.05     | krr     | oi_dft             |     3.5250 |    0.4727 |      0.0353 |     0.0047 |    0.8023 |
| phase3_armB_tau0.05     | krr     | pauli_dft          |     6.8157 |    0.6578 |      0.0337 |     0.0033 |    0.8277 |
| phase3_armB_tau0.05     | ridge   | cds_dft            |     0.4682 |    0.0359 |      0.0778 |     0.0060 |    0.3130 |
| phase3_armB_tau0.05     | ridge   | cpcm_dft           |     2.6755 |    0.1993 |      0.0588 |     0.0044 |    0.6677 |
| phase3_armB_tau0.05     | ridge   | d1_own             |     2.7047 |    0.1167 |      0.0715 |     0.0031 |    0.7400 |
| phase3_armB_tau0.05     | ridge   | d2_own             |     2.4728 |    0.1415 |      0.0516 |     0.0030 |    0.7798 |
| phase3_armB_tau0.05     | ridge   | disp_dft           |     0.0000 |    0.0000 |      0.0000 |     0.0000 |    1.0000 |
| phase3_armB_tau0.05     | ridge   | elst_dft           |     5.3015 |    0.3428 |      0.0522 |     0.0034 |    0.6945 |
| phase3_armB_tau0.05     | ridge   | oi_dft             |     4.0920 |    0.4578 |      0.0409 |     0.0046 |    0.7604 |
| phase3_armB_tau0.05     | ridge   | pauli_dft          |     8.1127 |    0.7981 |      0.0401 |     0.0039 |    0.7799 |
| phase3_armB_tau0.05     | svr     | cds_dft            |     0.3969 |    0.0324 |      0.0660 |     0.0054 |    0.4477 |
| phase3_armB_tau0.05     | svr     | cpcm_dft           |     2.5679 |    0.1725 |      0.0564 |     0.0038 |    0.6817 |
| phase3_armB_tau0.05     | svr     | d1_own             |     2.3323 |    0.0479 |      0.0617 |     0.0013 |    0.8368 |
| phase3_armB_tau0.05     | svr     | d2_own             |     2.2426 |    0.1557 |      0.0468 |     0.0032 |    0.8044 |
| phase3_armB_tau0.05     | svr     | disp_dft           |     0.0273 |    0.0041 |      0.0014 |     0.0002 |    0.9998 |
| phase3_armB_tau0.05     | svr     | elst_dft           |     5.0277 |    0.3814 |      0.0495 |     0.0038 |    0.7244 |
| phase3_armB_tau0.05     | svr     | oi_dft             |     3.6463 |    0.3250 |      0.0365 |     0.0032 |    0.7858 |
| phase3_armB_tau0.05     | svr     | pauli_dft          |     7.5618 |    0.8683 |      0.0374 |     0.0043 |    0.8002 |
| phase4_stacked_xgb_mace | stacked | cds_dft            |     0.4334 |    0.0283 |      0.0720 |     0.0047 |    0.3630 |
| phase4_stacked_xgb_mace | stacked | cpcm_dft           |     2.8913 |    0.1419 |      0.0635 |     0.0031 |    0.6161 |
| phase4_stacked_xgb_mace | stacked | d1_own             |     2.4118 |    0.1379 |      0.0638 |     0.0036 |    0.8298 |
| phase4_stacked_xgb_mace | stacked | d2_own             |     2.2482 |    0.1473 |      0.0469 |     0.0031 |    0.8128 |
| phase4_stacked_xgb_mace | stacked | disp_dft           |     0.0842 |    0.0293 |      0.0043 |     0.0015 |    0.9975 |
| phase4_stacked_xgb_mace | stacked | elst_dft           |     5.9646 |    0.5094 |      0.0587 |     0.0050 |    0.5529 |
| phase4_stacked_xgb_mace | stacked | oi_dft             |     5.1436 |    0.4295 |      0.0514 |     0.0043 |    0.6024 |
| phase4_stacked_xgb_mace | stacked | pauli_dft          |    10.5681 |    0.7760 |      0.0522 |     0.0038 |    0.5761 |
