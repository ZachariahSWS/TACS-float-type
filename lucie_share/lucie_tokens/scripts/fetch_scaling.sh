#!/usr/bin/env bash
# Pull the small per-run artefacts of the scaling sweep (metrics.jsonl,
# run_manifest.json, diag_val_*/diagnostics.json) from Bridges into a local
# mirror. tar over the ControlMaster's stdout works where scp does not.
#   scripts/fetch_scaling.sh [dest]   (default lucie/local/scaling/data)
set -uo pipefail
dest="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/local/scaling/data}"
mkdir -p "$dest"
timeout 300 ssh -o ControlPath="$HOME/.ssh/cm-%r@%h-%p" -o BatchMode=yes zschwab@bridges2.psc.edu \
  'cd $LUCIE_WORK/runs && tar cf - $(ls -d dynamics_era5t30_sc_r*_s* 2>/dev/null | while read d; do for f in $d/metrics.jsonl $d/run_manifest.json $d/diag_val_*/diagnostics.json $d/eval_ce_*.json; do [ -f "$f" ] && echo "$f"; done; done)' \
  | tar xf - -C "$dest"
echo "runs mirrored: $(ls -d "$dest"/dynamics_era5t30_sc_r* 2>/dev/null | wc -l)"
