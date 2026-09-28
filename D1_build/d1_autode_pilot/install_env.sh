#!/bin/bash
# install_env.sh — build the pilot's Python env on a compute node (SPEC §3.2 remedy, venv form).
#
#   sbatch -p <idle cpu partition> -o <D1_build>/logs/install_%j.log \
#          --export=ALL,D1_BUILD=<D1_build> install_env.sh
#
# What it builds: a venv at $D1_BUILD/env/d1ade layered on the `reactot` conda env
# (--system-site-packages), so rdkit / numpy / networkx / pandas / yaml / scipy are the
# exact reactot versions that built the D0 labels, while autodE lives only in the venv.
# reactot itself is never written to.
#   - Cython 3.1.8 (build-time only) from the wheel in $D1_BUILD/external/wheels (offline)
#   - autodE 1.4.5 from $D1_BUILD/external/autodE (git tag v1.4.5 = e7e71b33), compiled
#     extensions built with the node's C/C++ compiler, installed with --no-deps
# Idempotent: an existing venv is reused, pip reinstalls are no-ops for the same version.
#SBATCH --job-name=d1p_install
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G
set -euo pipefail
D1=${D1_BUILD:?D1_BUILD must point at the D1_build folder}
VENV=$D1/env/d1ade
SRC=$D1/external/autodE
WHL=$(ls "$D1"/external/wheels/cython-3.1.8-cp310-*.whl)
EXPECT_COMMIT=e7e71b33d8659fe921c16f877f9c1adddc4f9cff
echo "=== install_env $(date -Is) on $(hostname)"

source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
REACTOT_SP=$(python -c 'import site; print(site.getsitepackages()[0])')

got=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)
echo "autodE source: $SRC @ $got"
[ "$got" = "$EXPECT_COMMIT" ] || [ "$got" = unknown ] || { echo "STOP: autodE checkout is not v1.4.5 ($EXPECT_COMMIT)"; exit 1; }
command -v gcc g++ || { echo "STOP: no C/C++ compiler on this node"; exit 1; }
g++ --version | head -1

[ -x "$VENV/bin/python" ] || python -m venv --system-site-packages "$VENV"
source "$VENV/bin/activate"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
python -m pip install --no-index --no-deps "$WHL"
( cd "$SRC" && python -m pip install --no-index --no-deps --no-build-isolation . )

echo "=== verification"
cd "$(mktemp -d)"   # import from site-packages, not from the source tree
D1P_CONFIG=$D1/d1_autode_pilot/config.yaml REACTOT_SP=$REACTOT_SP VENV=$VENV \
python - <<'PY'
import importlib, os, sys
from pathlib import Path
print("python", sys.version.split()[0], sys.executable)
for m in ("autode", "rdkit", "networkx", "numpy", "pandas", "yaml", "scipy", "matplotlib", "Cython"):
    mod = importlib.import_module(m)
    print(f"  {m:10s} {getattr(mod, '__version__', '?'):12s} {Path(mod.__file__).parent}")
import autode
assert autode.__version__ == "1.4.5", autode.__version__
assert str(Path(autode.__file__).resolve()).startswith(str(Path(os.environ["VENV"]).resolve())), "autodE not from the venv"
assert not list(Path(os.environ["REACTOT_SP"]).glob("autode*")), "autodE leaked into reactot"
for ext in ("cconf_gen", "ade_dihedrals", "ade_rb_opt"):
    importlib.import_module(ext)
print("  compiled extensions: cconf_gen ade_dihedrals ade_rb_opt import OK")
from rdkit.Chem import rdDetermineBonds  # noqa: F401  (pilot needs RDKit >= 2022.09)

sys.path.insert(0, str(Path(os.environ["D1P_CONFIG"]).parent))
import pilot_common as pc
cfg = pc.load_config()
ade = pc.configure_autode(cfg, cores=8)
print("  ORCA available:", ade.methods.ORCA().is_available, " xtb available:", ade.methods.XTB().is_available)
for k, v in pc.hmethod_keywords_summary().items():
    print(f"  hmethod {k:8s}: {v!r}")
print("INSTALL_VERIFY PASS")
PY
"$(grep -E '^xtb_bin:' "$D1/d1_autode_pilot/config.yaml" | awk '{print $2}')" --version 2>&1 | grep -m1 -i "xtb version"
echo "=== done $(date -Is)"
