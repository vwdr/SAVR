#!/usr/bin/env bash
set -euo pipefail

project_root="/home/ved/SAVR"
run_id="acr-v5d-real-tensor-feasibility-v05"
run_root="${project_root}/results/${run_id}"
python_bin="${project_root}/envs/openvla-oft/bin/python"
manifest="${run_root}/launch/record.json"

if [[ "$(pwd -P)" != "${project_root}" ]]; then
  echo "Refusing V5-D v05 launch outside ${project_root}" >&2
  exit 1
fi
if [[ ! -x "${python_bin}" || ! -f "${manifest}" ]]; then
  echo "V5-D v05 pinned interpreter or launch manifest is missing" >&2
  exit 1
fi

physical_id="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_gpu"]["index"])' "${manifest}")"
selected_uuid="$(${python_bin} -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_gpu"]["uuid"])' "${manifest}")"

export SAVR_PHYSICAL_GPU_ID="${physical_id}"
export SAVR_SELECTED_GPU_UUID="${selected_uuid}"
export CUDA_VISIBLE_DEVICES="${physical_id}"
export HF_HOME="${run_root}/cache/huggingface"
export HF_HUB_CACHE="${run_root}/cache/huggingface/hub"
export TORCH_HOME="${run_root}/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${run_root}/cache/torchinductor"
export TRITON_CACHE_DIR="${run_root}/cache/triton"
export LIBERO_CONFIG_PATH="${run_root}/cache/libero"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${LIBERO_CONFIG_PATH}"

"${python_bin}" scripts/prepare_acr_v5_d_v05_libero_config.py

set +e
"${python_bin}" scripts/run_acr_v5_d_v05.py --backend torch-compile
status=$?
set -e
if [[ ${status} -eq 20 ]]; then
  "${python_bin}" scripts/run_acr_v5_d_v05.py --backend raw-cudagraph
  status=$?
fi
if [[ ${status} -ne 0 ]]; then
  exit "${status}"
fi
"${python_bin}" scripts/finalize_acr_v5_d_v05.py
