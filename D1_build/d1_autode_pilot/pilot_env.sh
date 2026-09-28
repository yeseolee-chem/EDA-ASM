# pilot_env.sh — sourced by run_job.sh / s0_preflight.sh / run_report.sh (compute nodes only).
# Environment block copied from label_true/scripts/smd_relabel_worker.sh (known-good ORCA 6.1.1 MPI setup).
PILOT_DIR=${PILOT_DIR:?PILOT_DIR must point at the d1_autode_pilot bundle directory}
CFG=${D1P_CONFIG:-$PILOT_DIR/config.yaml}
cfgget() { grep -E "^$1:" "$CFG" | head -1 | sed -E "s/^$1:[[:space:]]*//; s/[[:space:]]+#.*$//"; }
source "$(cfgget conda_sh)"
conda activate "$(cfgget conda_env)"
VENV=$(cfgget venv)
if [ -n "$VENV" ] && [ "$VENV" != null ]; then source "$VENV/bin/activate"; fi
export ORCA_BIN=$(cfgget orca_bin)
export XTB_BIN=$(cfgget xtb_bin)
export MPI_ROOT=$(cfgget mpi_root)
export SCRATCH=$(cfgget scratch)
export PATH=$MPI_ROOT/bin:$(dirname "$ORCA_BIN"):$(dirname "$XTB_BIN"):$PATH
export LD_LIBRARY_PATH=$MPI_ROOT/lib64:$(dirname "$ORCA_BIN"):${LD_LIBRARY_PATH:-}
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_pml=ob1
export OMPI_MCA_coll_hcoll_enable=0
export UCX_TLS=tcp,self,sm
ulimit -s unlimited 2>/dev/null || true
export OMP_STACKSIZE=4G
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$SCRATCH/jobs" "$SCRATCH/logs"

# run_stage JOB STAGE cmd... : skip if .done_STAGE, refuse if .fail_STAGE, time it, never retry a
# recorded failure (delete the .fail_ file by hand, with a reason in the log, to retry).
run_stage() {
    local job=$1 st=$2; shift 2
    local jd=$SCRATCH/jobs/$job; mkdir -p "$jd"
    if [ -f "$jd/.done_$st" ]; then echo "[$job] $st: done (skip)"; return 0; fi
    if [ -f "$jd/.fail_$st" ]; then echo "[$job] $st: FAILED earlier: $(cat "$jd/.fail_$st")"; return 1; fi
    echo "[$job] $st: start $(date -Is) on $(hostname)"
    local t0; t0=$(date +%s)
    ( cd "$PILOT_DIR" && "$@" ); local rc=$?
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$st" "$t0" "$(date +%s)" "$rc" "${SLURM_CPUS_PER_TASK:-1}" "$(hostname)" >> "$jd/timing.tsv"
    if [ -f "$jd/.done_$st" ]; then echo "[$job] $st: done"; return 0; fi
    echo "[$job] $st: no done marker (rc=$rc) — see log; a crash without .fail_$st is retried on resubmit"
    return 1
}

# run_sp JOB : the 5 single points in parallel (each ORCA process serial: eda.inp has %pal nprocs 1,
# frag*.inp have no %pal), as analysis/b3lyp_full/02_submit.sh "Option B"; idempotent per output.
run_sp() {
    local job=$1 spd=$SCRATCH/jobs/$1/sp
    [ -d "$spd" ] || { echo "no sp dir"; return 1; }
    cd "$spd" || return 1
    local pids=() s
    for s in eda frag1_dist frag2_dist frag1_rel frag2_rel; do
        if [ -f $s.out ] && grep -q "ORCA TERMINATED NORMALLY" $s.out; then continue; fi
        if [ $s = eda ]; then find . -maxdepth 1 -name 'eda*' ! -name 'eda.inp' -delete; else rm -f $s.out $s.err $s.gbw; fi
        ( "$ORCA_BIN" $s.inp > $s.out 2> $s.err ) &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done
    local bad=""
    for s in eda eda_frag1 eda_frag2 frag1_dist frag2_dist frag1_rel frag2_rel; do
        grep -q "ORCA TERMINATED NORMALLY" $s.out 2>/dev/null || bad="$bad $s"
    done
    if [ -z "$bad" ]; then
        rm -f ./*.tmp ./*.densities* 2>/dev/null
        date -Is > ../.done_sp
    else
        echo "ORCA did not terminate normally:$bad" > ../.fail_sp
    fi
}
