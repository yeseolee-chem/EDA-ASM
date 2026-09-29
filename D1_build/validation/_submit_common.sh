# _submit_common.sh — sourced by the submit_*.sh scripts (login node, one-shot; no python, no loops).
cd "$(dirname "$0")"
VAL=$PWD
PILOT=$(cd ../d1_autode_pilot && pwd)
SCR=$(grep -E '^scratch:' config_val.yaml | sed -E 's/^scratch:[[:space:]]*//; s/[[:space:]]+#.*$//')
mkdir -p "$SCR/logs"
need_slots() {   # $1 = tasks this submission adds; MaxSubmit=20 counts array tasks (squeue -r)
    local q; q=$(squeue -u "$USER" -h -r | wc -l)
    if [ $(( q + $1 )) -gt 20 ]; then echo "STOP: $q tasks queued; this submission needs $1 (MaxSubmit=20)"; exit 1; fi
    echo "queue before: $q"
}
# CLAUDE.md: whichever of cpu1 / cpu2 has the most idle CPUs
P=$(sinfo -h -p cpu1,cpu2 -o "%P %C" | tr -d '*' | awk '{split($2,a,"/"); if (a[2]>=m) {m=a[2]; p=$1}} END {print p}')
