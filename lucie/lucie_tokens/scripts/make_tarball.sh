#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-${project_dir}/../lucie_tokens.tar.gz}"
archive_name="$(basename "${project_dir}")"

# Package an explicit source allowlist. Generated data, environments, runs,
# checkpoints, build products, and arbitrary untracked files cannot enter it.
tar \
  --exclude="*/__pycache__" \
  --exclude="*.pyc" \
  --transform="s,^,${archive_name}/," \
  -czf "${destination}" \
  -C "${project_dir}" \
  .gitignore \
  README.md \
  pyproject.toml \
  requirements.lock \
  configs \
  docs \
  scripts \
  src/lucie_tokens \
  tests

destination_dir="$(dirname "${destination}")"
destination_name="$(basename "${destination}")"
(
  cd "${destination_dir}"
  sha256sum "${destination_name}"
) > "${destination}.sha256"
echo "Created ${destination}"
echo "Created ${destination}.sha256"
