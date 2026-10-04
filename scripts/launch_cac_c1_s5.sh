#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PHYSICAL_GPU_ID" >&2
  exit 64
fi

readonly PROJECT_ROOT="/home/ved/SAVR"
readonly CONFIG="configs/cac/c1_s5_requalification_v01.json"
readonly OUTPUT_ROOT="/home/ved/SAVR/results/cac-c1-s5-requalification-v01"
readonly RUNTIME="/home/ved/SAVR/envs/vla-cache-compat/bin/python"
readonly GPU_ID="$1"

cd "${PROJECT_ROOT}"
[[ ! -e "${OUTPUT_ROOT}" ]]
[[ -x "${RUNTIME}" ]]

CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  "${RUNTIME}" scripts/preflight_cac_c1_s5.py

IFS=',' read -r used_mib utilization_percent <<<"$(
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' '
)"
[[ "${used_mib}" =~ ^[0-9]+$ && "${utilization_percent}" =~ ^[0-9]+$ ]]
(( used_mib <= 1024 ))
(( utilization_percent <= 5 ))

exec timeout --signal=TERM --kill-after=60 3660 \
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    CAC_PHYSICAL_GPU_ID="${GPU_ID}" \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    "${RUNTIME}" scripts/run_cac_c1_worker.py \
      --config "${CONFIG}" \
      --output-root "${OUTPUT_ROOT}"
