#!/usr/bin/env bash
# Copy lucie/gust2_port (formerly lucie2) to ~/lucie2 on Bridges over the ssh ControlMaster socket (one cat
# per file, md5-verified). Usage: scripts/deploy.sh [files...]  (default: all .py/.sh/.md/.sbatch)
set -uo pipefail
host="${BRIDGES_HOST:-zschwab@bridges2.psc.edu}"
ssh_opts=(-o ControlPath="$HOME/.ssh/cm-%r@%h-%p")
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if [[ $# -eq 0 ]]; then
  mapfile -t files < <(find . -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name '*.sbatch' \) \
    -not -path './.venv/*' -not -path '*/__pycache__/*' | sed 's|^\./||' | sort)
else
  files=("$@")
fi
fail=0
for f in "${files[@]}"; do
  local_md5=$(md5sum "$f" | cut -d' ' -f1)
  if ! timeout 120 ssh "${ssh_opts[@]}" "$host" "mkdir -p \"\$(dirname lucie2/$f)\" && cat > lucie2/$f" < "$f"; then
    echo "COPY FAILED $f"; fail=1; continue
  fi
  remote_md5=$(timeout 60 ssh "${ssh_opts[@]}" "$host" "md5sum lucie2/$f" | cut -d' ' -f1)
  [[ "$local_md5" == "$remote_md5" ]] && echo "OK       $f" || { echo "MISMATCH $f"; fail=1; }
done
timeout 60 ssh "${ssh_opts[@]}" "$host" 'chmod +x lucie2/scripts/*.sh lucie2/*.py 2>/dev/null; mkdir -p lucie2/logs; true'
echo "deploy_fail=$fail"; exit $fail
