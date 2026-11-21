#!/usr/bin/env bash
# Local watcher for the scaling sweep on Bridges (run via the Monitor tool).
# Every 5 min over the ssh ControlMaster: emits one line per job state change
# (RUNNING once, every terminal state), per run's first logged epoch (loss
# finite?), per run reaching epoch 49, any non-finite loss, and any diag
# STAGE_EXIT != 0. Silence on the happy path between those events.
ssh_opts=(-o ControlPath="$HOME/.ssh/cm-%r@%h-%p" -o BatchMode=yes)
host=zschwab@bridges2.psc.edu
state=/tmp/scaling_monitor_state; touch "$state"
since="${SCALING_SINCE:-2026-09-21}"
fails=0
while true; do
  out=$(timeout 90 ssh "${ssh_opts[@]}" "$host" "sacct -S $since -u zschwab -n -X -P --format=JobID,JobName,State | grep -E '\|(diag_)?sc_r[0-9]+_s[0-9]+\|' | sed 's/ by [0-9]*//'; echo ---METRICS; cd \$LUCIE_WORK/runs && for d in dynamics_era5t30_sc_r*; do f=\$d/metrics.jsonl; [ -f \$f ] || continue; tail -1 \$f | python3 -c \"import sys,json; r=json.loads(sys.stdin.read()); print('\${d#dynamics_era5t30_}', r['epoch'], round(r['train_loss'],3), round(r['val_loss'],3), round(r['seconds']))\"; done; echo ---DIAG; grep -l 'STAGE_EXIT=[1-9]' ~/lucie_tokens/slurm-diag_sc_r*.out 2>/dev/null; grep -lE 'Traceback|Error' ~/lucie_tokens/slurm-sc_r*.out 2>/dev/null" 2>/dev/null)
  if [[ -z "$out" ]]; then
    fails=$((fails+1)); [[ $fails -ge 3 ]] && { echo "POLL FAILED x$fails (ssh socket down?)"; fails=0; }
    sleep 300; continue
  fi
  fails=0
  jobs=$(sed -n '1,/^---METRICS/p' <<<"$out" | grep -v '^---')
  metrics=$(sed -n '/^---METRICS/,/^---DIAG/p' <<<"$out" | grep -v '^---')
  diag=$(sed -n '/^---DIAG/,$p' <<<"$out" | grep -v '^---')
  # job states: emit on change, but never emit PENDING
  while IFS='|' read -r id name st; do
    [[ -z "$id" ]] && continue
    key="job:$id"; prev=$(grep "^$key=" "$state" | cut -d= -f2)
    [[ "$st" == "$prev" ]] && continue
    sed -i "/^$key=/d" "$state"; echo "$key=$st" >> "$state"
    [[ "$st" == "PENDING" ]] && continue
    echo "JOB $name ($id): $st"
  done <<<"$jobs"
  # metrics: first epoch, epoch 49, non-finite
  while read -r run ep tl vl sec; do
    [[ -z "$run" ]] && continue
    key="ep:$run"; prev=$(grep "^$key=" "$state" | cut -d= -f2)
    if [[ -z "$prev" ]]; then echo "FIRST $run epoch $ep train $tl val $vl ${sec}s/epoch"; fi
    if [[ "$ep" == "49" && "$prev" != "49" ]]; then echo "DONE  $run epoch 49 train $tl val $vl"; fi
    if [[ "$tl" == *nan* || "$vl" == *nan* || "$tl" == *inf* || "$vl" == *inf* ]]; then echo "NONFINITE $run epoch $ep train $tl val $vl"; fi
    sed -i "/^$key=/d" "$state"; echo "$key=$ep" >> "$state"
  done <<<"$metrics"
  while read -r f; do
    [[ -z "$f" ]] && continue
    key="diag:$f"; grep -q "^$key=" "$state" && continue
    echo "$key=1" >> "$state"; echo "ERRLOG $f"
  done <<<"$diag"
  sleep 300
done
