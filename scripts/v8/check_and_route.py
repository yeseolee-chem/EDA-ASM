"""Dispatcher: after ORCA jobs finish, either
  (a) diagnose failures + regen inputs + submit rerun + re-trigger self, OR
  (b) assemble labels + OOD if everything succeeded.

Max retries capped via env var CHECK_RETRY (default 3) to avoid infinite loop.
State: outputs/v8_review/labels/check_state.json  {"retry": N}
"""
from __future__ import annotations
import os, json, re, subprocess, sys
from pathlib import Path
import ase.io
from ase.data import chemical_symbols

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
SP_ROOT = V8 / "strain_sp"
MP = V8 / "manual_partitions.json"
STATE = V8 / "labels/check_state.json"
STATE.parent.mkdir(parents=True, exist_ok=True)
MAX_RETRY = int(os.environ.get("CHECK_RETRY", "3"))
LOG_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs")


def _fam(rid):
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def read_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"retry": 0}


def write_state(d):
    STATE.write_text(json.dumps(d, indent=1))


# --------- failure classification ---------

def classify_eda(rid):
    """Returns one of: None (done), 'not_run', 'scf', 'nocv', 'odd_electron', 'other'."""
    out = ORCA_ROOT / rid / "eda.out"
    if not out.exists():
        return "not_run"
    txt = out.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" in txt:
        return None
    if re.search(r'multiplicity \(\d+\) is odd and number of electrons \(\d+\) is odd', txt):
        return "odd_electron"
    if "failed in the EDA-NOCV" in txt:
        return "nocv"
    if ("DIIS Error" in txt or "MatrixLife" in txt or
        "This wavefunction IS NOT CONVERGED" in txt):
        return "scf"
    return "other"


def classify_sp(rid, frag):
    """SP failure classification. 'frag' in {fragA, fragB}."""
    out = SP_ROOT / rid / f"{frag}_R.out"
    if not out.exists():
        return "not_run"
    txt = out.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" in txt:
        return None
    if re.search(r'multiplicity \(\d+\) is odd and number of electrons \(\d+\) is odd', txt):
        return "odd_electron"
    if "DIIS Error" in txt or "MatrixLife" in txt or "This wavefunction IS NOT CONVERGED" in txt:
        return "scf"
    return "other"


def scan():
    m = json.loads(MP.read_text())
    eda_fails = {}   # rid -> reason
    sp_fails = {}    # (rid, frag) -> reason
    for rid in m:
        r = classify_eda(rid)
        if r:
            eda_fails[rid] = r
        for frag in ("fragA", "fragB"):
            r2 = classify_sp(rid, frag)
            if r2:
                sp_fails[(rid, frag)] = r2
    return eda_fails, sp_fails


# --------- input regen strategies ---------

def _atoms_line(sym, tag, x, y, z):
    return f"{sym}({tag})   {x:15.8f}   {y:15.8f}   {z:15.8f}"


def regen_eda_inp(rid, reason, retry_lvl):
    e = json.loads(MP.read_text()).get(rid, {})
    A = e.get("frag_A_indices", []); B = e.get("frag_B_indices", [])
    if not A or not B:
        return False
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers(); pos = ts_at.get_positions()
    frag_of = [None] * len(Z)
    for i in A: frag_of[i] = 1
    for i in B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        return False

    fam = _fam(rid)
    # Base charges
    tc = 0; fA_c = 0; fB_c = 0
    if fam in ("qmrxn20_sn2", "qmrxn20_e2"):
        fB_c = -1; tc = -1
    if reason == "odd_electron":
        # try charge=-1 on fragA (fragA usually anionic when its sumZ is odd)
        fA_c = -1; fB_c = 0; tc = -1

    keywords = "! BLYP D3BJ def2-TZVP NoSym EDA TightSCF"
    scf_extra = ""
    if reason in ("scf", "nocv") or retry_lvl >= 2:
        keywords = "! BLYP D3BJ def2-TZVP NoSym EDA VeryTightSCF SlowConv SOSCF defgrid3 NoTRAH"
        scf_extra = "%scf\n  MaxIter 500\n  DirectResetFreq 1\nend\n\n"
    frag_extra = ""
    if reason in ("scf", "nocv") or retry_lvl >= 2:
        frag_extra = " SlowConv SOSCF defgrid3"

    body = [keywords, "%maxcore 3500", ""]
    if scf_extra: body.append(scf_extra.rstrip())
    body += [
        "%eda",
        f'  FRAG1 "BLYP D3BJ def2-TZVP NoSym VeryTightSCF{frag_extra}"',
        f'  FRAG2 "BLYP D3BJ def2-TZVP NoSym VeryTightSCF{frag_extra}"',
        f"  FRAG1_C {fA_c}",
        "  FRAG1_M 1",
        f"  FRAG2_C {fB_c}",
        "  FRAG2_M 1",
        "end", "",
        f"* xyz {tc} 1",
    ]
    for i in range(len(Z)):
        body.append(_atoms_line(chemical_symbols[int(Z[i])], frag_of[i], pos[i,0], pos[i,1], pos[i,2]))
    body += ["*", ""]
    (ORCA_ROOT / rid / "eda.inp").write_text("\n".join(body))
    (ORCA_ROOT / rid / "eda.out").unlink(missing_ok=True)
    (ORCA_ROOT / rid / "eda.err").unlink(missing_ok=True)
    return True


def regen_sp_inp(rid, frag, reason, retry_lvl):
    e = json.loads(MP.read_text()).get(rid, {})
    key = "frag_A_indices_R" if frag == "fragA" else "frag_B_indices_R"
    frag_idx = e.get(key, [])
    if not frag_idx:
        return False
    r_at = ase.io.read(str(RAW / rid / "R.xyz"))
    Z = r_at.get_atomic_numbers(); pos = r_at.get_positions()
    fam = _fam(rid)
    charge = 0
    if fam in ("qmrxn20_sn2", "qmrxn20_e2") and frag == "fragB":
        charge = -1
    if reason == "odd_electron":
        charge = -1  # try fragment charged
    keywords = "! BLYP D3BJ def2-TZVP NoSym TightSCF"
    if reason == "scf" or retry_lvl >= 2:
        keywords = "! BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3"
    lines = [keywords, "%maxcore 3500", "", f"* xyz {charge} 1"]
    for i in frag_idx:
        lines.append(f"{chemical_symbols[int(Z[i])]}   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines += ["*", ""]
    d = SP_ROOT / rid; d.mkdir(exist_ok=True)
    (d / f"{frag}_R.inp").write_text("\n".join(lines))
    (d / f"{frag}_R.out").unlink(missing_ok=True)
    (d / f"{frag}_R.err").unlink(missing_ok=True)
    return True


# --------- submit runners ---------

def submit_rerun(eda_rids, sp_pairs, retry_lvl):
    """Write manifests + submit sbatch runner. Returns job id."""
    manifest_eda = ORCA_ROOT / "manifest_retry.txt"
    manifest_sp = SP_ROOT / "manifest_retry.txt"
    manifest_eda.write_text("\n".join(eda_rids) + "\n" if eda_rids else "")
    manifest_sp.write_text("\n".join(f"{r} {f}" for r, f in sp_pairs) + "\n" if sp_pairs else "")

    script = REPO / "scripts/v8/rerun_retry.sh"
    if not script.exists():
        write_rerun_script(script)
    out = subprocess.check_output(["sbatch", "--parsable", str(script)]).decode().strip()
    return out.split(";")[0]


def write_rerun_script(path):
    path.write_text(RETRY_SCRIPT_TEMPLATE)
    path.chmod(0o755)


RETRY_SCRIPT_TEMPLATE = r"""#!/bin/bash
#SBATCH --job-name=orca_retry
#SBATCH --partition=cpu2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_retry.%j.out
#SBATCH --error=/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/orca_retry.%j.err
set -uo pipefail
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot
[ -f "$HOME/orca6/orca-env.sh" ] && source "$HOME/orca6/orca-env.sh"

REPO=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction
ORCA_BIN=$HOME/orca_6_1_1_avx2/orca

# EDA rerun
if [ -s "$REPO/outputs/v8_review/orca_inputs/manifest_retry.txt" ]; then
  while IFS= read -r RID; do
    [ -z "$RID" ] && continue
    DIR="$REPO/outputs/v8_review/orca_inputs/$RID"
    [ -f "$DIR/eda.inp" ] || continue
    echo "[$(date +%H:%M:%S)] EDA START $RID"
    (cd "$DIR" && "$ORCA_BIN" eda.inp > eda.out 2> eda.err)
    if grep -q "ORCA TERMINATED NORMALLY" "$DIR/eda.out" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] EDA OK    $RID"
      (cd "$DIR" && rm -f *.densities *.gbw *.bas* *.tmp *.smpso *.smpss *.opt *.hess *.engrad 2>/dev/null)
    else
      echo "[$(date +%H:%M:%S)] EDA FAIL  $RID"
    fi
  done < "$REPO/outputs/v8_review/orca_inputs/manifest_retry.txt"
fi

# SP rerun
if [ -s "$REPO/outputs/v8_review/strain_sp/manifest_retry.txt" ]; then
  while IFS= read -r LINE; do
    [ -z "$LINE" ] && continue
    RID=$(echo "$LINE" | awk '{print $1}')
    FRAG=$(echo "$LINE" | awk '{print $2}')
    DIR="$REPO/outputs/v8_review/strain_sp/$RID"
    [ -f "$DIR/${FRAG}_R.inp" ] || continue
    echo "[$(date +%H:%M:%S)] SP START $RID $FRAG"
    (cd "$DIR" && "$ORCA_BIN" "${FRAG}_R.inp" > "${FRAG}_R.out" 2> "${FRAG}_R.err")
    if grep -q "ORCA TERMINATED NORMALLY" "$DIR/${FRAG}_R.out" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] SP OK    $RID $FRAG"
      (cd "$DIR" && rm -f ${FRAG}_R.densities ${FRAG}_R.gbw ${FRAG}_R.bas* ${FRAG}_R.tmp 2>/dev/null)
    else
      echo "[$(date +%H:%M:%S)] SP FAIL  $RID $FRAG"
    fi
  done < "$REPO/outputs/v8_review/strain_sp/manifest_retry.txt"
fi
echo "[$(date +%H:%M:%S)] retry done"
"""


def submit_trigger(dep_id):
    """Submit a trigger that reruns this script after dep_id."""
    script = REPO / "scripts/v8/trigger_finalize.sh"
    out = subprocess.check_output(
        ["sbatch", f"--dependency=afterany:{dep_id}", "--parsable", str(script)]
    ).decode().strip()
    return out.split(";")[0]


# --------- finalize ---------

def finalize():
    """Run assemble_labels + detect_ood."""
    print("[FINALIZE] running assemble_labels_v8.py")
    subprocess.check_call([sys.executable, str(REPO / "scripts/v8/assemble_labels_v8.py")])
    print("[FINALIZE] running detect_ood_v8.py")
    subprocess.check_call([sys.executable, str(REPO / "scripts/v8/detect_ood_v8.py")])
    print("[FINALIZE] DONE.")


# --------- main ---------

def main():
    state = read_state()
    retry = state.get("retry", 0)
    print(f"[check_and_route] retry={retry}/{MAX_RETRY}")

    eda_fails, sp_fails = scan()
    print(f"[check_and_route] EDA fails: {len(eda_fails)}  SP fails: {len(sp_fails)}")
    for rid, r in list(eda_fails.items())[:20]:
        print(f"  EDA {rid}  reason={r}")
    for (rid, frag), r in list(sp_fails.items())[:20]:
        print(f"  SP  {rid} {frag}  reason={r}")

    if not eda_fails and not sp_fails:
        finalize()
        return

    if retry >= MAX_RETRY:
        print(f"[check_and_route] hit max retry ({MAX_RETRY}). finalizing with partial data.")
        finalize()
        return

    # Regen inputs
    eda_rids = []
    for rid, reason in eda_fails.items():
        if reason == "not_run":
            eda_rids.append(rid); continue
        if regen_eda_inp(rid, reason, retry):
            eda_rids.append(rid)
    sp_pairs = []
    for (rid, frag), reason in sp_fails.items():
        if reason == "not_run":
            sp_pairs.append((rid, frag)); continue
        if regen_sp_inp(rid, frag, reason, retry):
            sp_pairs.append((rid, frag))

    if not eda_rids and not sp_pairs:
        print("[check_and_route] nothing to submit. finalize.")
        finalize(); return

    print(f"[check_and_route] submitting rerun for EDA={len(eda_rids)}  SP={len(sp_pairs)}")
    rerun_id = submit_rerun(eda_rids, sp_pairs, retry)
    print(f"[check_and_route] rerun job: {rerun_id}")

    trig_id = submit_trigger(rerun_id)
    print(f"[check_and_route] next trigger job: {trig_id}")

    state["retry"] = retry + 1
    write_state(state)


if __name__ == "__main__":
    main()
