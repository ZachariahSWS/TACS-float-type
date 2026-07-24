#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${1:-${project_dir}/../lucie_tokens.tar.gz}"
archive_root="$(dirname "${project_dir}")"
archive_name="$(basename "${project_dir}")"

tar \
  --exclude="${archive_name}/.venv" \
  --exclude="${archive_name}/data" \
  --exclude="${archive_name}/runs" \
  --exclude="${archive_name}/profiles" \
  --exclude="*/__pycache__" \
  --exclude="*/.pytest_cache" \
  --exclude="*.egg-info" \
  --exclude="*.pyc" \
  -czf "${destination}" \
  -C "${archive_root}" \
  "${archive_name}"

sha256sum "${destination}" > "${destination}.sha256"
echo "Created ${destination}"
echo "Created ${destination}.sha256"
