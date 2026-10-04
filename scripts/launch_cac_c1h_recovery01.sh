#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="/home/ved/SAVR"
readonly CONFIG="configs/cac/c1h_headroom_v02_recovery01.json"
readonly OUTPUT_ROOT="/home/ved/SAVR/results/cac-c1h-headroom-v02-recovery01"
readonly RUNTIME="/home/ved/SAVR/envs/vla-cache-compat/bin/python"
readonly GPU_ID="${CAC_SELECTED_GPU_ID:?CAC_SELECTED_GPU_ID must name one preselected physical GPU}"

[[ "${GPU_ID}" =~ ^[0-9]+$ ]]
cd "${PROJECT_ROOT}"
[[ ! -e "${OUTPUT_ROOT}" ]]
[[ -x "${RUNTIME}" ]]

IFS=',' read -r used_mib utilization_percent <<<"$(
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)"
[[ "${used_mib}" =~ ^[0-9]+$ && "${utilization_percent}" =~ ^[0-9]+$ ]]
(( used_mib <= 1024 ))
(( utilization_percent <= 5 ))

exec timeout --signal=TERM --kill-after=60 36120 \
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    CAC_PHYSICAL_GPU_ID="${GPU_ID}" \
    LIBERO_CONFIG_PATH="${PROJECT_ROOT}/configs/pair/libero_runtime" \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    "${RUNTIME}" "${PROJECT_ROOT}/scripts/run_cac_c1h_worker.py" \
      --config "${CONFIG}" \
      --output-root "${OUTPUT_ROOT}"
