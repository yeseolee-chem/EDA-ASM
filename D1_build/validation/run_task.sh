#!/bin/bash
# run_task.sh TASK_ID — one validation task inside a worker allocation (called by worker.sh).
#   exit 0  finished: the last stage is done, or a stage recorded a scientific failure (.fail_<stage>)
#   exit 1  infrastructure failure: crash without a marker, missing input, unknown task
# tasks.csv columns: task_id,exp,priority,type,job_id,stages,arg   (stages separated by ';')
set -uo pipefail
T=${1:?task id}
VAL_DIR=${VAL_DIR:?}
export PILOT_DIR=${PILOT_DIR:-$(cd "$VAL_DIR/../d1_autode_pilot" && pwd)}
export D1P_CONFIG=$VAL_DIR/config_val.yaml
export D1P_MANIFEST=$VAL_DIR/val_manifest.csv
source "$PILOT_DIR/pilot_env.sh"                 # run_stage / run_sp are shell functions: source, not inherit
export VAL_QUEUE=${VAL_QUEUE:-$SCRATCH/queue}
row=$(awk -F, -v t="$T" 'NR>1 && $1==t {print; exit}' "$VAL_DIR/tasks.csv")
[ -n "$row" ] || { echo "no task $T in tasks.csv"; exit 1; }
IFS=, read -r TID EXP PRIO TYPE JOB STAGES ARG <<< "$row"
echo "=== $TID ($EXP, priority $PRIO, $TYPE) $(date -Is) $(hostname) ==="
cd "$PILOT_DIR" || exit 1

case $TYPE in
chain)
    jd=$SCRATCH/jobs/$JOB
    if [ "$EXP" = "V6c" ] && [ ! -d "$jd/ts" ]; then
        # V6c: the pilot's autodE tree (checkpoints) is COPIED; the pilot scratch stays untouched and
        # the copy carries no .fail_ts. ade_name keeps the autodE checkpoint hash (see pilot_common.ade_name).
        src=$(cfgget pilot_scratch)/jobs/$ARG/ts
        [ -d "$src" ] || { echo "V6c: $src missing"; exit 1; }
        mkdir -p "$jd" && cp -a "$src" "$jd/" || exit 1
        echo "V6c: copied $src -> $jd/ts"
    fi
    for st in ${STAGES//;/ }; do
        case $st in
            ts)     cmd=(python run_ts.py --job "$JOB") ;;
            ref)    cmd=(python make_reference.py --job "$JOB") ;;
            inputs) cmd=(python build_sp_inputs.py --job "$JOB") ;;
            sp)     cmd=(run_sp "$JOB") ;;
            label)  cmd=(python assemble.py --job "$JOB") ;;
            refopt) cmd=(python refopt.py --job "$JOB") ;;
            *) echo "unknown stage $st"; exit 1 ;;
        esac
        run_stage "$JOB" "$st" "${cmd[@]}" || break
    done
    last=${STAGES##*;}
    [ -f "$jd/.done_$last" ] && exit 0
    compgen -G "$jd/.fail_*" > /dev/null && { echo "scientific failure: $(cat "$jd"/.fail_*)"; exit 0; }
    exit 1 ;;
v1a|v1b) python "$VAL_DIR/val_determinism.py" --mode "$TYPE" --src "$ARG" --task "$TID" ;;
v2a)     python "$VAL_DIR/val_grad.py" --rxn "$ARG" --task "$TID" ;;
v7)      python "$VAL_DIR/val_scan_j03.py" --src "$ARG" --task "$TID" ;;
v8)      python "$VAL_DIR/analysis/negative_controls.py" --task "$TID" ;;
v9)      python "$VAL_DIR/analysis/gate_audit_d0.py" --task "$TID" ;;
*)       echo "unknown task type $TYPE"; exit 1 ;;
esac
