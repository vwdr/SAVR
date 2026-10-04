#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PHYSICAL_GPU_ID" >&2
  exit 64
fi

project_root=/home/ved/SAVR
cd "$project_root"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  envs/vla-cache-compat/bin/python scripts/preflight_openvla_d62_requalification_s4.py

exec env \
  CUDA_VISIBLE_DEVICES="$1" \
  OPENVLA_PHYSICAL_GPU_ID="$1" \
  PYTHONPATH=src \
  envs/vla-cache-compat/bin/python scripts/run_openvla_d62_requalification_s4.py \
  --config configs/openvla/d62_requalification_s4_v01.json
