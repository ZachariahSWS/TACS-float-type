#!/usr/bin/env bash
# Copy project files to ~/lucie_tokens on Bridges-2 over the shared ssh
# ControlMaster socket, one `cat` per file (scp fails over the multiplexed
# socket), and verify each md5. Paths are relative to the project root.
#
#   scripts/deploy_to_bridges.sh src/lucie_tokens/training.py configs/x.toml ...
set -uo pipefail
host="${BRIDGES_HOST:-zschwab@bridges2.psc.edu}"
ssh_opts=(-o ControlPath="$HOME/.ssh/cm-%r@%h-%p")
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
fail=0
for f in "$@"; do
  if [[ ! -f "$f" ]]; then echo "MISSING(local) $f"; fail=1; continue; fi
  local_md5=$(md5sum "$f" | cut -d' ' -f1)
  if ! timeout 120 ssh "${ssh_opts[@]}" "$host" "mkdir -p \"\$(dirname lucie_tokens/$f)\" && cat > lucie_tokens/$f" < "$f"; then
    echo "COPY FAILED $f"; fail=1; continue
  fi
  remote_md5=$(timeout 60 ssh "${ssh_opts[@]}" "$host" "md5sum lucie_tokens/$f" | cut -d' ' -f1)
  if [[ "$local_md5" == "$remote_md5" ]]; then echo "OK       $f"; else echo "MISMATCH $f"; fail=1; fi
done
timeout 60 ssh "${ssh_opts[@]}" "$host" 'chmod +x lucie_tokens/scripts/*.sh lucie_tokens/scripts/*.py 2>/dev/null; true'
echo "deploy_fail=$fail"
exit $fail
