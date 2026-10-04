#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 PHYSICAL_GPU_ID" >&2
  exit 64
fi

project_root=/home/ved/SAVR
config_path=configs/openvla/d62_requalification_s4_recovery01_v01.json
cd "$project_root"

CUDA_VISIBLE_DEVICES="" PYTHONPATH=src \
  envs/vla-cache-compat/bin/python \
  scripts/preflight_openvla_d62_requalification_s4_recovery01.py

config_sha256="$(${project_root}/envs/vla-cache-compat/bin/python -c \
  'import json; from pathlib import Path; print(json.loads(Path("configs/openvla/d62_requalification_s4_recovery01_v01.json").read_text())["semantic_sha256"])')"

CUDA_VISIBLE_DEVICES="$1" OPENVLA_PHYSICAL_GPU_ID="$1" PYTHONPATH=src \
  envs/vla-cache-compat/bin/python \
  scripts/run_openvla_compact_position_qualification.py \
  --config-sha256 "$config_sha256"

exec env \
  CUDA_VISIBLE_DEVICES="$1" \
  OPENVLA_PHYSICAL_GPU_ID="$1" \
  PYTHONPATH=src \
  envs/vla-cache-compat/bin/python \
  scripts/run_openvla_d62_requalification_s4_recovery01.py \
  --config "$config_path"
