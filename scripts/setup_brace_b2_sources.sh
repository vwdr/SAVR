#!/usr/bin/env bash
set -euo pipefail

expected_root=/home/ved/SAVR
if [[ "$PWD" != "$expected_root" ]]; then
  echo "Refusing BRACE-B2 source setup outside $expected_root" >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "BRACE-B2 source setup refuses visible CUDA devices" >&2
  exit 1
fi

clone_sparse() {
  local url=$1
  local target=$2
  local revision=$3
  shift 3
  if [[ ! -d "$target/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$target"
    git -C "$target" sparse-checkout init --cone
  fi
  git -C "$target" fetch --filter=blob:none origin "$revision"
  git -C "$target" sparse-checkout set "$@"
  git -C "$target" checkout --detach "$revision"
  test -z "$(git -C "$target" status --porcelain)"
}

clone_sparse \
  https://github.com/chen7086/VLA-ADP.git \
  third_party/vla-adp \
  d7094b09a4996847772c1fa975f09d863e1b759a \
  experiments prismatic scripts

clone_sparse \
  https://github.com/MINT-SJTU/VLA-Pruner.git \
  third_party/vla-pruner \
  84d4b7192c77abf1585610e2f12393319b7ebff9 \
  src

clone_sparse \
  https://github.com/alexwhz-sjtu/SpecPrune-VLA.git \
  third_party/specprune-vla \
  8091adc4b574ce9008d49a1dc9a210f4eec314c1 \
  openvla-oft

du -sb third_party/vla-adp third_party/vla-pruner third_party/specprune-vla
