#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PHYSICAL_GPU_ID" >&2
  exit 64
fi

project_root=/home/ved/SAVR
cd "$project_root"

exec env \
  CUDA_VISIBLE_DEVICES="$1" \
  OPENVLA_PHYSICAL_GPU_ID="$1" \
  envs/vla-cache-compat/bin/python \
  scripts/run_openvla_semantic_parity_recovery01.py \
  --config configs/openvla/semantic_parity_s3_v02_recovery01.json
