# HP Tuning Reproduction: Our vs Espley

- Our models: 36
- Espley models: 30

| Model | Param | Espley | Ours | Match |
|---|---|---|---|---|
| `2_st_nn_distortion_energy_1_dft` | `dropout_rate_1` | 0.2 | 0.2 | ✓ |
| `2_st_nn_distortion_energy_1_dft` | `dropout_rate_2` | 0.2 | 0.2 | ✓ |
| `2_st_nn_distortion_energy_1_dft` | `learning_rate` | 0.001 | 0.001 | ✓ |
| `2_st_nn_distortion_energy_1_dft` | `neurons_1` | 64 | 256 | ❌ |
| `2_st_nn_distortion_energy_1_dft` | `neurons_2` | 32 | 128 | ❌ |
| `2_st_nn_distortion_energy_1_dft` | `reg_val_1` | 0.001 | 0.001 | ✓ |
| `2_st_nn_distortion_energy_1_dft` | `reg_val_2` | 0.001 | 0.001 | ✓ |
| `2_st_nn_distortion_energy_2_dft` | `dropout_rate_1` | 0.2 | 0.2 | ✓ |
| `2_st_nn_distortion_energy_2_dft` | `dropout_rate_2` | 0.2 | 0.2 | ✓ |
| `2_st_nn_distortion_energy_2_dft` | `learning_rate` | 0.001 | 0.001 | ✓ |
| `2_st_nn_distortion_energy_2_dft` | `neurons_1` | 256 | 64 | ❌ |
| `2_st_nn_distortion_energy_2_dft` | `neurons_2` | 128 | 32 | ❌ |
| `2_st_nn_distortion_energy_2_dft` | `reg_val_1` | 0.001 | 0.001 | ✓ |
| `2_st_nn_distortion_energy_2_dft` | `reg_val_2` | 0.001 | 0.001 | ✓ |
| `2_st_nn_e_barrier_dft` | `dropout_rate_1` | 0.2 | 0.2 | ✓ |
| `2_st_nn_e_barrier_dft` | `dropout_rate_2` | 0.2 | 0.2 | ✓ |
| `2_st_nn_e_barrier_dft` | `learning_rate` | 0.001 | 0.001 | ✓ |
| `2_st_nn_e_barrier_dft` | `neurons_1` | 256 | 64 | ❌ |
| `2_st_nn_e_barrier_dft` | `neurons_2` | 128 | 32 | ❌ |
| `2_st_nn_e_barrier_dft` | `reg_val_1` | 0.001 | 0.001 | ✓ |
| `2_st_nn_e_barrier_dft` | `reg_val_2` | 0.001 | 0.001 | ✓ |
| `2_st_nn_interaction_energies_dft` | `dropout_rate_1` | 0.2 | 0.2 | ✓ |
| `2_st_nn_interaction_energies_dft` | `dropout_rate_2` | 0.2 | 0.2 | ✓ |
| `2_st_nn_interaction_energies_dft` | `learning_rate` | 0.001 | 0.001 | ✓ |
| `2_st_nn_interaction_energies_dft` | `neurons_1` | 256 | 256 | ✓ |
| `2_st_nn_interaction_energies_dft` | `neurons_2` | 128 | 128 | ✓ |
| `2_st_nn_interaction_energies_dft` | `reg_val_1` | 0.001 | 0.001 | ✓ |
| `2_st_nn_interaction_energies_dft` | `reg_val_2` | 0.001 | 0.001 | ✓ |
| `2_st_nn_q_barrier_dft` | `dropout_rate_1` | 0.2 | 0.2 | ✓ |
| `2_st_nn_q_barrier_dft` | `dropout_rate_2` | 0.2 | 0.2 | ✓ |
| `2_st_nn_q_barrier_dft` | `learning_rate` | 0.001 | 0.001 | ✓ |
| `2_st_nn_q_barrier_dft` | `neurons_1` | 64 | 64 | ✓ |
| `2_st_nn_q_barrier_dft` | `neurons_2` | 32 | 32 | ✓ |
| `2_st_nn_q_barrier_dft` | `reg_val_1` | 0.001 | 0.001 | ✓ |
| `2_st_nn_q_barrier_dft` | `reg_val_2` | 0.001 | 0.001 | ✓ |
| `2_st_nn_sum_distortion_energies_dft` | `dropout_rate_1` | 0.2 | 0.2 | ✓ |
| `2_st_nn_sum_distortion_energies_dft` | `dropout_rate_2` | 0.2 | 0.2 | ✓ |
| `2_st_nn_sum_distortion_energies_dft` | `learning_rate` | 0.001 | 0.001 | ✓ |
| `2_st_nn_sum_distortion_energies_dft` | `neurons_1` | 64 | 64 | ✓ |
| `2_st_nn_sum_distortion_energies_dft` | `neurons_2` | 32 | 32 | ✓ |
| `2_st_nn_sum_distortion_energies_dft` | `reg_val_1` | 0.001 | 0.001 | ✓ |
| `2_st_nn_sum_distortion_energies_dft` | `reg_val_2` | 0.001 | 0.001 | ✓ |
| `4_st_nn_distortion_energy_1_dft` | `dropout_rate_1` | 0.1 | 0.1 | ✓ |
| `4_st_nn_distortion_energy_1_dft` | `dropout_rate_2` | 0.6 | 0.1 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `dropout_rate_3` | 0.2 | 0.2 | ✓ |
| `4_st_nn_distortion_energy_1_dft` | `dropout_rate_4` | 0.5 | 0.1 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `learning_rate` | 0.0001 | 0.0001 | ✓ |
| `4_st_nn_distortion_energy_1_dft` | `neurons_1` | 256 | 128 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `neurons_2` | 512 | 32 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `neurons_3` | 512 | 64 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `neurons_4` | 128 | 512 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `reg_val_1` | 0.01 | 0.001 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `reg_val_2` | 0.01 | 0.001 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `reg_val_3` | 0.001 | 0.01 | ❌ |
| `4_st_nn_distortion_energy_1_dft` | `reg_val_4` | 0.001 | 0.001 | ✓ |
| `4_st_nn_distortion_energy_2_dft` | `dropout_rate_1` | 0.1 | 0.1 | ✓ |
| `4_st_nn_distortion_energy_2_dft` | `dropout_rate_2` | 0.6 | 0.1 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `dropout_rate_3` | 0.2 | 0.2 | ✓ |
| `4_st_nn_distortion_energy_2_dft` | `dropout_rate_4` | 0.5 | 0.1 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `learning_rate` | 0.0001 | 0.0001 | ✓ |
| `4_st_nn_distortion_energy_2_dft` | `neurons_1` | 256 | 128 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `neurons_2` | 512 | 32 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `neurons_3` | 512 | 64 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `neurons_4` | 128 | 512 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `reg_val_1` | 0.01 | 0.001 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `reg_val_2` | 0.01 | 0.001 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `reg_val_3` | 0.001 | 0.01 | ❌ |
| `4_st_nn_distortion_energy_2_dft` | `reg_val_4` | 0.001 | 0.001 | ✓ |
| `4_st_nn_e_barrier_dft` | `dropout_rate_1` | 0.1 | 0.1 | ✓ |
| `4_st_nn_e_barrier_dft` | `dropout_rate_2` | 0.6 | 0.1 | ❌ |
| `4_st_nn_e_barrier_dft` | `dropout_rate_3` | 0.2 | 0.2 | ✓ |
| `4_st_nn_e_barrier_dft` | `dropout_rate_4` | 0.5 | 0.1 | ❌ |
| `4_st_nn_e_barrier_dft` | `learning_rate` | 0.0001 | 0.0001 | ✓ |
| `4_st_nn_e_barrier_dft` | `neurons_1` | 256 | 128 | ❌ |
| `4_st_nn_e_barrier_dft` | `neurons_2` | 512 | 32 | ❌ |
| `4_st_nn_e_barrier_dft` | `neurons_3` | 512 | 64 | ❌ |
| `4_st_nn_e_barrier_dft` | `neurons_4` | 128 | 512 | ❌ |
| `4_st_nn_e_barrier_dft` | `reg_val_1` | 0.01 | 0.001 | ❌ |
| `4_st_nn_e_barrier_dft` | `reg_val_2` | 0.01 | 0.001 | ❌ |
| `4_st_nn_e_barrier_dft` | `reg_val_3` | 0.001 | 0.01 | ❌ |
| `4_st_nn_e_barrier_dft` | `reg_val_4` | 0.001 | 0.001 | ✓ |
| `4_st_nn_interaction_energies_dft` | `dropout_rate_1` | 0.1 | 0.1 | ✓ |
| `4_st_nn_interaction_energies_dft` | `dropout_rate_2` | 0.6 | 0.1 | ❌ |
| `4_st_nn_interaction_energies_dft` | `dropout_rate_3` | 0.2 | 0.2 | ✓ |
| `4_st_nn_interaction_energies_dft` | `dropout_rate_4` | 0.5 | 0.1 | ❌ |
| `4_st_nn_interaction_energies_dft` | `learning_rate` | 0.0001 | 0.0001 | ✓ |
| `4_st_nn_interaction_energies_dft` | `neurons_1` | 256 | 128 | ❌ |
| `4_st_nn_interaction_energies_dft` | `neurons_2` | 512 | 32 | ❌ |
| `4_st_nn_interaction_energies_dft` | `neurons_3` | 512 | 64 | ❌ |
| `4_st_nn_interaction_energies_dft` | `neurons_4` | 128 | 512 | ❌ |
| `4_st_nn_interaction_energies_dft` | `reg_val_1` | 0.01 | 0.001 | ❌ |
| `4_st_nn_interaction_energies_dft` | `reg_val_2` | 0.01 | 0.001 | ❌ |
| `4_st_nn_interaction_energies_dft` | `reg_val_3` | 0.001 | 0.01 | ❌ |
| `4_st_nn_interaction_energies_dft` | `reg_val_4` | 0.001 | 0.001 | ✓ |
| `4_st_nn_q_barrier_dft` | `dropout_rate_1` | 0.1 | 0.1 | ✓ |
| `4_st_nn_q_barrier_dft` | `dropout_rate_2` | 0.1 | 0.1 | ✓ |
| `4_st_nn_q_barrier_dft` | `dropout_rate_3` | 0.2 | 0.2 | ✓ |
| `4_st_nn_q_barrier_dft` | `dropout_rate_4` | 0.1 | 0.1 | ✓ |
| `4_st_nn_q_barrier_dft` | `learning_rate` | 0.0001 | 0.0001 | ✓ |
| `4_st_nn_q_barrier_dft` | `neurons_1` | 128 | 128 | ✓ |
| `4_st_nn_q_barrier_dft` | `neurons_2` | 32 | 32 | ✓ |
| `4_st_nn_q_barrier_dft` | `neurons_3` | 64 | 64 | ✓ |
| `4_st_nn_q_barrier_dft` | `neurons_4` | 512 | 512 | ✓ |
| `4_st_nn_q_barrier_dft` | `reg_val_1` | 0.001 | 0.001 | ✓ |
| `4_st_nn_q_barrier_dft` | `reg_val_2` | 0.001 | 0.001 | ✓ |
| `4_st_nn_q_barrier_dft` | `reg_val_3` | 0.01 | 0.01 | ✓ |
| `4_st_nn_q_barrier_dft` | `reg_val_4` | 0.001 | 0.001 | ✓ |
| `4_st_nn_sum_distortion_energies_dft` | `dropout_rate_1` | 0.1 | 0.1 | ✓ |
| `4_st_nn_sum_distortion_energies_dft` | `dropout_rate_2` | 0.6 | 0.1 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `dropout_rate_3` | 0.2 | 0.2 | ✓ |
| `4_st_nn_sum_distortion_energies_dft` | `dropout_rate_4` | 0.5 | 0.1 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `learning_rate` | 0.0001 | 0.0001 | ✓ |
| `4_st_nn_sum_distortion_energies_dft` | `neurons_1` | 256 | 128 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `neurons_2` | 512 | 32 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `neurons_3` | 512 | 64 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `neurons_4` | 128 | 512 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `reg_val_1` | 0.01 | 0.001 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `reg_val_2` | 0.01 | 0.001 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `reg_val_3` | 0.001 | 0.01 | ❌ |
| `4_st_nn_sum_distortion_energies_dft` | `reg_val_4` | 0.001 | 0.001 | ✓ |
| `krr_distortion_energy_1_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `krr_distortion_energy_1_dft` | `gamma` | None | None | ✓ |
| `krr_distortion_energy_2_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `krr_distortion_energy_2_dft` | `gamma` | None | None | ✓ |
| `krr_e_barrier_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `krr_e_barrier_dft` | `gamma` | None | None | ✓ |
| `krr_interaction_energies_dft` | `alpha` | 0.6 | 0.6 | ✓ |
| `krr_interaction_energies_dft` | `gamma` | None | None | ✓ |
| `krr_q_barrier_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `krr_q_barrier_dft` | `gamma` | None | None | ✓ |
| `krr_sum_distortion_energies_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `krr_sum_distortion_energies_dft` | `gamma` | None | None | ✓ |
| `ridge_distortion_energy_1_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `ridge_distortion_energy_1_dft` | `tol` | 0.1 | 0.1 | ✓ |
| `ridge_distortion_energy_2_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `ridge_distortion_energy_2_dft` | `tol` | 0.1 | 0.1 | ✓ |
| `ridge_e_barrier_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `ridge_e_barrier_dft` | `tol` | 0.1 | 0.1 | ✓ |
| `ridge_interaction_energies_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `ridge_interaction_energies_dft` | `tol` | 0.1 | 0.1 | ✓ |
| `ridge_q_barrier_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `ridge_q_barrier_dft` | `tol` | 0.1 | 0.1 | ✓ |
| `ridge_sum_distortion_energies_dft` | `alpha` | 1.0 | 1.0 | ✓ |
| `ridge_sum_distortion_energies_dft` | `tol` | 0.1 | 0.1 | ✓ |
| `svr_distortion_energy_1_dft` | `C` | 30 | 30 | ✓ |
| `svr_distortion_energy_1_dft` | `coef0` | 0 | 0 | ✓ |
| `svr_distortion_energy_1_dft` | `degree` | 1 | 1 | ✓ |
| `svr_distortion_energy_1_dft` | `epsilon` | 1 | 1 | ✓ |
| `svr_distortion_energy_1_dft` | `gamma` | auto | auto | ✓ |
| `svr_distortion_energy_2_dft` | `C` | 30 | 30 | ✓ |
| `svr_distortion_energy_2_dft` | `coef0` | 0 | 0 | ✓ |
| `svr_distortion_energy_2_dft` | `degree` | 1 | 1 | ✓ |
| `svr_distortion_energy_2_dft` | `epsilon` | 0.25 | 0.25 | ✓ |
| `svr_distortion_energy_2_dft` | `gamma` | scale | scale | ✓ |
| `svr_e_barrier_dft` | `C` | 50 | 50 | ✓ |
| `svr_e_barrier_dft` | `coef0` | 0 | 0 | ✓ |
| `svr_e_barrier_dft` | `degree` | 1 | 1 | ✓ |
| `svr_e_barrier_dft` | `epsilon` | 1 | 1 | ✓ |
| `svr_e_barrier_dft` | `gamma` | auto | auto | ✓ |
| `svr_interaction_energies_dft` | `C` | 50 | 50 | ✓ |
| `svr_interaction_energies_dft` | `coef0` | 0 | 0 | ✓ |
| `svr_interaction_energies_dft` | `degree` | 1 | 1 | ✓ |
| `svr_interaction_energies_dft` | `epsilon` | 1 | 1 | ✓ |
| `svr_interaction_energies_dft` | `gamma` | scale | scale | ✓ |
| `svr_q_barrier_dft` | `C` | 50 | 50 | ✓ |
| `svr_q_barrier_dft` | `coef0` | 0 | 0 | ✓ |
| `svr_q_barrier_dft` | `degree` | 1 | 1 | ✓ |
| `svr_q_barrier_dft` | `epsilon` | 1 | 1 | ✓ |
| `svr_q_barrier_dft` | `gamma` | auto | auto | ✓ |
| `svr_sum_distortion_energies_dft` | `C` | 50 | 50 | ✓ |
| `svr_sum_distortion_energies_dft` | `coef0` | 0 | 0 | ✓ |
| `svr_sum_distortion_energies_dft` | `degree` | 1 | 1 | ✓ |
| `svr_sum_distortion_energies_dft` | `epsilon` | 1 | 1 | ✓ |
| `svr_sum_distortion_energies_dft` | `gamma` | auto | auto | ✓ |

## Summary
- Espley models: 30
- Ours has 6 extras: ['rf_q_barrier_dft', 'rf_e_barrier_dft', 'rf_sum_distortion_energies_dft', 'rf_interaction_energies_dft', 'rf_distortion_energy_1_dft', 'rf_distortion_energy_2_dft']
- Missing in ours: 0
- Parameter matches: **123**
- Parameter mismatches: **51**
- Match rate: **70.7%**

## Per-model match rate
| Model | Match | Mismatch |
|---|---|---|
| `2_st_nn_distortion_energy_1_dft` | 5 | 2 |
| `2_st_nn_distortion_energy_2_dft` | 5 | 2 |
| `2_st_nn_e_barrier_dft` | 5 | 2 |
| `2_st_nn_interaction_energies_dft` | 7 | 0 |
| `2_st_nn_q_barrier_dft` | 7 | 0 |
| `2_st_nn_sum_distortion_energies_dft` | 7 | 0 |
| `4_st_nn_distortion_energy_1_dft` | 4 | 9 |
| `4_st_nn_distortion_energy_2_dft` | 4 | 9 |
| `4_st_nn_e_barrier_dft` | 4 | 9 |
| `4_st_nn_interaction_energies_dft` | 4 | 9 |
| `4_st_nn_q_barrier_dft` | 13 | 0 |
| `4_st_nn_sum_distortion_energies_dft` | 4 | 9 |
| `krr_distortion_energy_1_dft` | 2 | 0 |
| `krr_distortion_energy_2_dft` | 2 | 0 |
| `krr_e_barrier_dft` | 2 | 0 |
| `krr_interaction_energies_dft` | 2 | 0 |
| `krr_q_barrier_dft` | 2 | 0 |
| `krr_sum_distortion_energies_dft` | 2 | 0 |
| `ridge_distortion_energy_1_dft` | 2 | 0 |
| `ridge_distortion_energy_2_dft` | 2 | 0 |
| `ridge_e_barrier_dft` | 2 | 0 |
| `ridge_interaction_energies_dft` | 2 | 0 |
| `ridge_q_barrier_dft` | 2 | 0 |
| `ridge_sum_distortion_energies_dft` | 2 | 0 |
| `svr_distortion_energy_1_dft` | 5 | 0 |
| `svr_distortion_energy_2_dft` | 5 | 0 |
| `svr_e_barrier_dft` | 5 | 0 |
| `svr_interaction_energies_dft` | 5 | 0 |
| `svr_q_barrier_dft` | 5 | 0 |
| `svr_sum_distortion_energies_dft` | 5 | 0 |