#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${LUCIE_IMAGE:-/ocean/containers/ngc/pytorch/pytorch_24.11-py3.sif}"
dependencies="${LUCIE_DEPS:-${HOME}/lucie_deps_py312}"

if [[ ! -r "${image}" ]]; then
  echo "Container is not readable: ${image}" >&2
  exit 2
fi
if [[ ! -d "${dependencies}" ]]; then
  echo "Dependency directory is missing: ${dependencies}" >&2
  exit 2
fi
if ! command -v apptainer >/dev/null 2>&1; then
  echo "apptainer is not available" >&2
  exit 2
fi

container_pythonpath="${project_dir}/src:${dependencies}"
export APPTAINERENV_PYTHONPATH="${container_pythonpath}"
for name in LUCIE_WORK LUCIE_SOURCE LUCIE_RUN LUCIE_SEED; do
  if [[ -n "${!name:-}" ]]; then
    export "APPTAINERENV_${name}=${!name}"
  fi
done

apptainer_options=(exec --bind /ocean:/ocean --bind /jet:/jet)
if [[ "${LUCIE_USE_GPU:-1}" == "1" ]]; then
  apptainer_options+=(--nv)
fi

exec apptainer "${apptainer_options[@]}" \
  "${image}" \
  python -m lucie_tokens "$@"
