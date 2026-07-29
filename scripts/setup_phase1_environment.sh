#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_ROOT="/home/ved/SAVR"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly MAMBA_ROOT="${PROJECT_ROOT}/.micromamba"
readonly MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"
readonly ENV_PREFIX="${PROJECT_ROOT}/envs/openvla-oft"
readonly THIRD_PARTY="${PROJECT_ROOT}/third_party"
readonly CACHE_ROOT="${PROJECT_ROOT}/cache"
readonly LOCK_DIR="${PROJECT_ROOT}/environment/locks"

readonly MICROMAMBA_VERSION="2.6.2-1"
readonly MICROMAMBA_SHA256="e9683b483df06dbd3fdd8a37f1b6826d7e5caf4e85bf15a0af4fbad3d4ad1a58"
readonly OPENVLA_OFT_COMMIT="e4287e94541f459edc4feabc4e181f537cd569a8"
readonly LIBERO_COMMIT="8f1084e3132a39270c3a13ebe37270a43ece2a01"
readonly TRANSFORMERS_COMMIT="bc339d9ad707454c0c115970db43c260067c61ab"
readonly DLIMP_COMMIT="040105d256bd28866cc6620621a3d5f7b6b91b46"

if [[ "${PROJECT_ROOT}" != "${EXPECTED_ROOT}" ]]; then
  echo "Refusing to run outside ${EXPECTED_ROOT}: ${PROJECT_ROOT}" >&2
  exit 2
fi

mkdir -p \
  "${MAMBA_ROOT}/bin" \
  "${THIRD_PARTY}" \
  "${CACHE_ROOT}/pip" \
  "${CACHE_ROOT}/huggingface" \
  "${CACHE_ROOT}/torch" \
  "${LOCK_DIR}"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  curl -fL \
    "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64" \
    -o "${MAMBA_BIN}"
  printf '%s  %s\n' "${MICROMAMBA_SHA256}" "${MAMBA_BIN}" | sha256sum --check -
  chmod 755 "${MAMBA_BIN}"
fi

clone_at_commit() {
  local repository_url="$1"
  local commit="$2"
  local destination="$3"

  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${repository_url}" "${destination}"
  fi
  git -C "${destination}" fetch --depth=1 origin "${commit}"
  git -C "${destination}" checkout --detach "${commit}"
  test "$(git -C "${destination}" rev-parse HEAD)" = "${commit}"
}

clone_at_commit \
  "https://github.com/moojink/openvla-oft.git" \
  "${OPENVLA_OFT_COMMIT}" \
  "${THIRD_PARTY}/openvla-oft"
clone_at_commit \
  "https://github.com/Lifelong-Robot-Learning/LIBERO.git" \
  "${LIBERO_COMMIT}" \
  "${THIRD_PARTY}/LIBERO"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  "${MAMBA_BIN}" create \
    --yes \
    --root-prefix "${MAMBA_ROOT}" \
    --prefix "${ENV_PREFIX}" \
    --file "${PROJECT_ROOT}/environment/phase1-conda.yml"
else
  "${MAMBA_BIN}" install \
    --yes \
    --root-prefix "${MAMBA_ROOT}" \
    --prefix "${ENV_PREFIX}" \
    --file "${PROJECT_ROOT}/environment/phase1-conda.yml"
fi

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export PYTHONNOUSERSITE=1

run_python() {
  "${MAMBA_BIN}" run --root-prefix "${MAMBA_ROOT}" --prefix "${ENV_PREFIX}" "$@"
}

run_python python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0

run_python python -m pip install \
  "numpy<2" \
  "accelerate>=0.25.0" \
  draccus==0.8.0 \
  einops \
  huggingface_hub \
  json-numpy \
  jsonlines \
  matplotlib \
  peft==0.11.1 \
  protobuf \
  rich \
  sentencepiece==0.1.99 \
  timm==0.9.10 \
  tokenizers==0.19.1 \
  wandb \
  tensorflow==2.15.0 \
  tensorflow_datasets==4.9.3 \
  tensorflow_graphics==2021.12.3 \
  diffusers==0.30.3 \
  imageio \
  uvicorn \
  fastapi \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@${TRANSFORMERS_COMMIT}" \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla.git@${DLIMP_COMMIT}"

run_python python -m pip install \
  numpy==1.26.4 \
  "imageio[ffmpeg]" \
  robosuite==1.4.1 \
  mujoco==2.3.7 \
  opencv-python==4.6.0.66 \
  bddl==1.0.1 \
  easydict==1.9 \
  cloudpickle==2.1.0 \
  gym==0.25.2

run_python python -m pip install \
  --no-deps \
  --config-settings editable_mode=compat \
  --editable "${THIRD_PARTY}/LIBERO"
run_python python -m pip install --no-deps --editable "${THIRD_PARTY}/openvla-oft"
run_python python -m pip check

"${MAMBA_BIN}" list \
  --root-prefix "${MAMBA_ROOT}" \
  --prefix "${ENV_PREFIX}" \
  --explicit > "${LOCK_DIR}/conda-linux-64-explicit.txt"
run_python python -m pip freeze --all > "${LOCK_DIR}/pip-freeze.txt"

printf 'Phase 1 environment installation completed at %s\n' "${ENV_PREFIX}"
