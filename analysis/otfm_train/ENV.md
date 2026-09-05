# reactot conda env — pinned versions for otfm_train

react-ot's source code (v0.1.0 from `github.com/deepprinciple/react-ot`,
main branch commit `6dfccd0`) is written against **PyTorch Lightning 1.x**
API despite its `env.yaml` pinning `pytorch-lightning==2.4.0` (upstream
inconsistency). Attempting to run under PL 2.4.0 fails with a cascade of
API removals:

  - `Trainer(replace_sampler_ddp=...)` — renamed to `use_distributed_sampler`
  - `Trainer(strategy=None)` — must be `"auto"` or a Strategy instance
  - `SBModule.training_epoch_end / validation_epoch_end` — removed;
    replaced by `on_train_epoch_end` etc. with different signature
    (outputs stored on `self` instead of passed as arg)

Six methods in `reactot/trainer/pl_trainer.py` and
`reactot/trainer/potential_module.py` use the removed `*_epoch_end`
API. Refactoring them properly requires editing `training_step` too
(store outputs on self, clear in epoch-end hook). That's fragile.

## Chosen fix

Downgrade PyTorch Lightning to **1.9.5** in the `reactot` conda env,
which is the last major-version release before PL 2.0 removed the APIs
react-ot uses. Reversed all PL 2.x-specific regex patches from
`06_train_crossfit.py` / `08_train_final.py` PATCH_RULES; kept the
colored_traceback softening (version-agnostic).

## Commands used (reproducible)

```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate reactot
pip install --no-deps "pytorch-lightning==1.9.5"
pip install "setuptools<81"   # PL 1.9 imports pkg_resources; removed in setuptools 81+
```

## Verified state

```
pytorch-lightning  1.9.5
setuptools         80.10.2
torch              2.2.1+cu118   (unchanged; PL 1.9.5 works with torch 2.x)
ase                3.28.0        (patched in _rot_patches.py for ase.mep.neb)
```

Test that survived:

```python
from reactot.trainer.pl_trainer import SBModule, DDPMModule
from pytorch_lightning import Trainer
Trainer(max_epochs=1, accelerator='cpu', strategy=None, devices=1,
        replace_sampler_ddp=False)
# → OK (both kwargs valid in PL 1.9)
```

## If env gets rebuilt

Re-run the two `pip install` commands above. `env.yaml` in the react-ot
repo is intentionally NOT the source of truth for us — it pins the PL
version that fails.
