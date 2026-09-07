# SPEC17rev2 schema probe

## Shipped checkpoint (pre-patch)
### pretrained checkpoint
top-level keys: ['epoch', 'global_step', 'pytorch-lightning_version', 'state_dict', 'loops', 'callbacks', 'optimizer_states', 'lr_schedulers', 'hparams_name', 'hyper_parameters']
  hp.model_config: {'pos_require_grad': False, 'cutoff': 10.0, 'num_layers': 6, 'hidden_channels': 196, 'num_radial': 96, 'in_hidden_channels': 8, 'reflect_equiv': True, 'legacy': True, 'update': True, 'pos_grad': False
  hp.optimizer_config: {'lr': 0}
  hp.training_config: {'datadir': 'reactot/data/transition1x/', 'remove_h': False, 'bz': 14, 'num_workers': 0, 'clip_grad': True, 'gradient_clip_val': None, 'ema': False, 'ema_decay': 0.999, 'swapping_react_prod': True, 'a
  hp.node_nfs: [9, 9, 9]
  hp.edge_nf: 0
  hp.condition_nf: 1
  hp.fragment_names: ['R', 'TS', 'P']
  hp.pos_dim: 3
  hp.update_pocket_coords: True
  hp.condition_time: True
  hp.edge_cutoff: None
  hp.norm_values: (1.0, 1.0, 1.0)
  hp.norm_biases: (0.0, 0.0, 0.0)
  hp.noise_schedule: cosine
  hp.timesteps: 3000
  hp.precision: 1e-05
  hp.loss_type: l2
  hp.pos_only: True
  hp.process_type: TS1x
  hp.model: <class 'reactot.model.leftnet.LEFTNet'>
  hp.enforce_same_encoding: None
  hp.scales: [1.0, 2.0, 1.0]
  hp.eval_epochs: 1
  hp.source: None
  hp.fixed_idx: [0, 2]
  hp.mapping: R+P->TS
  hp.mapping_initial: RP
  hp.beta_max: 0.3
  hp.nfe: 25
  hp.ot_ode: True
  hp.power: 0.5
  hp.inv_power: 1
  hp.sigma: 0.0
  hp.ts_guess: None
  hp.idx: 1
  hp.pbc: False

state_dict param count: 246
atom-count-sensitive layers (embed / one_hot / encoder / decoder):
  ddpm.dynamics.model.embedding.weight: (196, 8)
  ddpm.dynamics.model.embedding.bias: (196,)
  ddpm.dynamics.model.embedding_out.weight: (8, 196)
  ddpm.dynamics.model.embedding_out.bias: (8,)
  ddpm.dynamics.model.neighbor_emb.embedding.weight: (196, 8)
  ddpm.dynamics.model.neighbor_emb.embedding.bias: (196,)
  ddpm.dynamics.model.distance_embedding.mlp.0.linear.weight: (98, 96)
  ddpm.dynamics.model.distance_embedding.mlp.1.linear.weight: (196, 98)
  ddpm.dynamics.encoders.0.mlp.0.linear.weight: (12, 6)
  ddpm.dynamics.encoders.0.mlp.0.linear.bias: (12,)
  ddpm.dynamics.encoders.0.mlp.1.linear.weight: (6, 12)
  ddpm.dynamics.encoders.0.mlp.1.linear.bias: (6,)
  ddpm.dynamics.encoders.1.mlp.0.linear.weight: (12, 6)
  ddpm.dynamics.encoders.1.mlp.0.linear.bias: (12,)
  ddpm.dynamics.encoders.1.mlp.1.linear.weight: (6, 12)
  ddpm.dynamics.encoders.1.mlp.1.linear.bias: (6,)
  ddpm.dynamics.encoders.2.mlp.0.linear.weight: (12, 6)
  ddpm.dynamics.encoders.2.mlp.0.linear.bias: (12,)
  ddpm.dynamics.encoders.2.mlp.1.linear.weight: (6, 12)
  ddpm.dynamics.encoders.2.mlp.1.linear.bias: (6,)
  ddpm.dynamics.decoders.0.mlp.0.linear.weight: (12, 6)
  ddpm.dynamics.decoders.0.mlp.0.linear.bias: (12,)
  ddpm.dynamics.decoders.0.mlp.1.linear.weight: (6, 12)
  ddpm.dynamics.decoders.0.mlp.1.linear.bias: (6,)
  ddpm.dynamics.decoders.1.mlp.0.linear.weight: (12, 6)
  ddpm.dynamics.decoders.1.mlp.0.linear.bias: (12,)
  ddpm.dynamics.decoders.1.mlp.1.linear.weight: (6, 12)
  ddpm.dynamics.decoders.1.mlp.1.linear.bias: (6,)
  ddpm.dynamics.decoders.2.mlp.0.linear.weight: (12, 6)
  ddpm.dynamics.decoders.2.mlp.0.linear.bias: (12,)
  ddpm.dynamics.decoders.2.mlp.1.linear.weight: (6, 12)
  ddpm.dynamics.decoders.2.mlp.1.linear.bias: (6,)

## Expected reshape after GATE-6b patch
`ATOM_MAPPING` widens 5 → 7 elements (adds Cl, Br). `node_nfs` moves from `[9]*3` to `[11]*3`. Embed/output layers listed above become the partial_load skip set.

### train_rpsb_all.pkl
top-level keys: ['reactant', 'transition_state', 'product', 'single_fragment', 'use_ind', 'ts_guess', 'ts_guess_sbv1', 'ts_guess_true', 'ts_guess_NEBCI-xtb']
  reactant: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', 'wB97x_6-31G(d).energy', 'wB97x_6-31G(d).atomization_energy', 'wB97x_6-31G(d).forces', 'formula', 'xtb_positions']
     num_atoms: len=10073  sample=7
     charges: len=10073  sample=[8, 6, 7, 7, 6, 1, 1]
     fragments: len=10073  sample=[[0, 1, 2, 3, 4, 5, 6]]
     positions: len=10073  sample=[[ 0.06342524 -0.92886364  0.22975714]
 [ 1.0350014  -0.0944876  -0.20364845]
 [
     rxn: len=10073  sample=rxn2091
     wB97x_6-31G(d).energy: len=10073  sample=-7130.09805147626
     wB97x_6-31G(d).atomization_energy: len=10073  sample=-32.44572913700813
     wB97x_6-31G(d).forces: len=10073  sample=[[-0.014290707539290617, 0.020040185772169136, -0.026752898103443237], [-0.01700
     formula: len=10073  sample=C2H2N2O
     xtb_positions: len=10073  sample=[[ 0.06152958 -0.93377726  0.22518693]
 [ 1.03139108 -0.10755434 -0.20176287]
 [
  transition_state: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', 'wB97x_6-31G(d).energy', 'wB97x_6-31G(d).atomization_energy', 'wB97x_6-31G(d).forces', 'formula']
     num_atoms: len=10073  sample=7
     charges: len=10073  sample=[8, 6, 7, 7, 6, 1, 1]
     fragments: len=10073  sample=[[0, 1, 2, 3, 4, 5, 6]]
     positions: len=10073  sample=[[ 0.29101042 -1.33406032 -0.15308241]
 [ 0.98165975 -0.30763056 -0.16231558]
 [
     rxn: len=10073  sample=rxn2091
     wB97x_6-31G(d).energy: len=10073  sample=-7126.253932088064
     wB97x_6-31G(d).atomization_energy: len=10073  sample=-28.601609748812734
     wB97x_6-31G(d).forces: len=10073  sample=[[0.004967984837668417, -0.010599632077756587, -0.006827900898302871], [-0.00775
     formula: len=10073  sample=C2H2N2O
  product: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', 'wB97x_6-31G(d).energy', 'wB97x_6-31G(d).atomization_energy', 'wB97x_6-31G(d).forces', 'formula', 'xtb_positions']
     num_atoms: len=10073  sample=7
     charges: len=10073  sample=[8, 6, 7, 7, 6, 1, 1]
     fragments: len=10073  sample=[[0, 1, 2, 3, 4, 5, 6]]
     positions: len=10073  sample=[[ 1.81454067 -0.942652    0.81290956]
 [ 1.0633961  -0.37792317  0.0728469 ]
 [
     rxn: len=10073  sample=rxn2091
     wB97x_6-31G(d).energy: len=10073  sample=-7127.571511082589
     wB97x_6-31G(d).atomization_energy: len=10073  sample=-29.919188743337145
     wB97x_6-31G(d).forces: len=10073  sample=[[-0.018530797025030417, -0.027001602782120757, -0.0003693072180399153], [-0.030
     formula: len=10073  sample=C2H2N2O
     xtb_positions: len=10073  sample=[[ 1.36244483 -1.2570311   0.75618552]
 [ 1.11984092 -0.3690715  -0.01300002]
 [
  single_fragment: len=10073  sample=1
  use_ind: len=9000  sample=0
  ts_guess: len=10073  sample=[[ 0.88215103 -1.11903842  0.71113736]
 [ 1.05110039 -0.16726637 -0.07710477]
 [
  ts_guess_sbv1: len=10073  sample=[[ 0.33669215 -1.2871399   0.07352323]
 [ 1.0093096  -0.26497617 -0.10088502]
 [
  ts_guess_true: len=10073  sample=[[ 0.29592142 -1.319564   -0.12497216]
 [ 0.9865708  -0.29313427 -0.13420534]
 [
  ts_guess_NEBCI-xtb: len=10073  sample=[[ 0.63894857 -1.19362143  0.61105   ]
 [ 1.09324857 -0.23766143  0.02559   ]
 [

### valid_rpsb_all.pkl
top-level keys: ['reactant', 'transition_state', 'product', 'single_fragment', 'use_ind', 'ts_guess', 'ts_guess_sbv1', 'ts_guess_true', 'ts_guess_NEBCI-xtb']
  reactant: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', 'wB97x_6-31G(d).energy', 'wB97x_6-31G(d).atomization_energy', 'wB97x_6-31G(d).forces', 'formula', 'xtb_positions']
     num_atoms: len=10073  sample=7
     charges: len=10073  sample=[8, 6, 7, 7, 6, 1, 1]
     fragments: len=10073  sample=[[0, 1, 2, 3, 4, 5, 6]]
     positions: len=10073  sample=[[ 0.06342524 -0.92886364  0.22975714]
 [ 1.0350014  -0.0944876  -0.20364845]
 [
     rxn: len=10073  sample=rxn2091
     wB97x_6-31G(d).energy: len=10073  sample=-7130.09805147626
     wB97x_6-31G(d).atomization_energy: len=10073  sample=-32.44572913700813
     wB97x_6-31G(d).forces: len=10073  sample=[[-0.014290707539290617, 0.020040185772169136, -0.026752898103443237], [-0.01700
     formula: len=10073  sample=C2H2N2O
     xtb_positions: len=10073  sample=[[ 0.06152958 -0.93377726  0.22518693]
 [ 1.03139108 -0.10755434 -0.20176287]
 [
  transition_state: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', 'wB97x_6-31G(d).energy', 'wB97x_6-31G(d).atomization_energy', 'wB97x_6-31G(d).forces', 'formula']
     num_atoms: len=10073  sample=7
     charges: len=10073  sample=[8, 6, 7, 7, 6, 1, 1]
     fragments: len=10073  sample=[[0, 1, 2, 3, 4, 5, 6]]
     positions: len=10073  sample=[[ 0.29101042 -1.33406032 -0.15308241]
 [ 0.98165975 -0.30763056 -0.16231558]
 [
     rxn: len=10073  sample=rxn2091
     wB97x_6-31G(d).energy: len=10073  sample=-7126.253932088064
     wB97x_6-31G(d).atomization_energy: len=10073  sample=-28.601609748812734
     wB97x_6-31G(d).forces: len=10073  sample=[[0.004967984837668417, -0.010599632077756587, -0.006827900898302871], [-0.00775
     formula: len=10073  sample=C2H2N2O
  product: ['num_atoms', 'charges', 'fragments', 'positions', 'rxn', 'wB97x_6-31G(d).energy', 'wB97x_6-31G(d).atomization_energy', 'wB97x_6-31G(d).forces', 'formula', 'xtb_positions']
     num_atoms: len=10073  sample=7
     charges: len=10073  sample=[8, 6, 7, 7, 6, 1, 1]
     fragments: len=10073  sample=[[0, 1, 2, 3, 4, 5, 6]]
     positions: len=10073  sample=[[ 1.81454067 -0.942652    0.81290956]
 [ 1.0633961  -0.37792317  0.0728469 ]
 [
     rxn: len=10073  sample=rxn2091
     wB97x_6-31G(d).energy: len=10073  sample=-7127.571511082589
     wB97x_6-31G(d).atomization_energy: len=10073  sample=-29.919188743337145
     wB97x_6-31G(d).forces: len=10073  sample=[[-0.018530797025030417, -0.027001602782120757, -0.0003693072180399153], [-0.030
     formula: len=10073  sample=C2H2N2O
     xtb_positions: len=10073  sample=[[ 1.36244483 -1.2570311   0.75618552]
 [ 1.11984092 -0.3690715  -0.01300002]
 [
  single_fragment: len=10073  sample=1
  use_ind: len=1073  sample=7
  ts_guess: len=10073  sample=[[ 0.88215103 -1.11903842  0.71113736]
 [ 1.05110039 -0.16726637 -0.07710477]
 [
  ts_guess_sbv1: len=10073  sample=[[ 0.33669215 -1.2871399   0.07352323]
 [ 1.0093096  -0.26497617 -0.10088502]
 [
  ts_guess_true: len=10073  sample=[[ 0.29592142 -1.319564   -0.12497216]
 [ 0.9865708  -0.29313427 -0.13420534]
 [
  ts_guess_NEBCI-xtb: len=10073  sample=[[ 0.63894857 -1.19362143  0.61105   ]
 [ 1.09324857 -0.23766143  0.02559   ]
 [

