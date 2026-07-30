#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_ROOT="/home/ved/SAVR"
readonly VLA_CACHE_REVISION="a4909880573868dee2769343d52e793c0341678b"
readonly TRANSFORMERS_REVISION="9a90a37acacf453433168db8d7769b7ea3c40c06"
readonly SOURCE_ROOT="${EXPECTED_ROOT}/third_party/vla-cache"
readonly CORE_PYTHON="${EXPECTED_ROOT}/envs/openvla-oft/bin/python"
readonly COMPAT_ENV="${EXPECTED_ROOT}/envs/vla-cache-compat"
readonly PIP_CACHE="${EXPECTED_ROOT}/cache/pip-vla-cache"

export PIP_CACHE_DIR="${PIP_CACHE}"
export PIP_DISABLE_PIP_VERSION_CHECK="1"

if [[ "$(pwd -P)" != "${EXPECTED_ROOT}" ]]; then
  echo "Refusing to run outside ${EXPECTED_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${CORE_PYTHON}" ]]; then
  echo "Validated core Python is unavailable: ${CORE_PYTHON}" >&2
  exit 2
fi

mkdir -p \
  "${EXPECTED_ROOT}/third_party" \
  "${EXPECTED_ROOT}/envs" \
  "${PIP_CACHE}"
if [[ ! -d "${SOURCE_ROOT}/.git" ]]; then
  git clone --filter=blob:none https://github.com/siyuhsu/vla-cache.git "${SOURCE_ROOT}"
fi
git -C "${SOURCE_ROOT}" fetch --depth 1 origin "${VLA_CACHE_REVISION}"
git -C "${SOURCE_ROOT}" checkout --detach "${VLA_CACHE_REVISION}"
if [[ -n "$(git -C "${SOURCE_ROOT}" status --porcelain)" ]]; then
  echo "Pinned VLA-Cache source is dirty" >&2
  exit 2
fi

if [[ ! -x "${COMPAT_ENV}/bin/python" ]]; then
  "${CORE_PYTHON}" -m venv --system-site-packages "${COMPAT_ENV}"
fi

"${COMPAT_ENV}/bin/python" -m pip install --no-deps \
  "tokenizers==0.21.1" \
  "scikit-image==0.25.0" \
  "seaborn==0.13.2" \
  "lazy-loader==0.4" \
  "tifffile==2025.5.10" \
  "pandas==2.2.3" \
  "pytz==2025.2" \
  "tzdata==2025.2"
"${COMPAT_ENV}/bin/python" -m pip install --no-deps \
  "transformers @ git+https://github.com/siyuhsu/transformers.git@${TRANSFORMERS_REVISION}"

"${COMPAT_ENV}/bin/python" - <<'PY'
import tokenizers
import transformers
import skimage
import seaborn
import pandas

assert tokenizers.__version__ == "0.21.1"
assert transformers.__version__ == "4.47.0"
assert skimage.__version__ == "0.25.0"
assert seaborn.__version__ == "0.13.2"
assert pandas.__version__ == "2.2.3"
print("Pinned isolated VLA-Cache compatibility environment is ready.")
PY

git -C "${SOURCE_ROOT}" rev-parse HEAD
du -sh "${SOURCE_ROOT}" "${COMPAT_ENV}"
