#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-tts-eval-chatterbox-turbo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
APP_DIR="${REPO_ROOT}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_CACHE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}"
DEFAULT_REFS_DIR="${REPO_ROOT}/data/refs"
DEVICE="${CHATTERBOX_TURBO_DEVICE:-cuda}"
TEMPERATURE="${CHATTERBOX_TURBO_TEMPERATURE:-0.8}"
TOP_P="${CHATTERBOX_TURBO_TOP_P:-0.95}"
TOP_K="${CHATTERBOX_TURBO_TOP_K:-1000}"
REPETITION_PENALTY="${CHATTERBOX_TURBO_REPETITION_PENALTY:-1.2}"
OUTPUT_DIR="/app/data/inputs/chatterbox_turbo"

if [[ $# -gt 2 ]]; then
  echo "usage: d.sh [refs_dir] [timestamp]" >&2
  exit 1
fi

REFS_DIR="$(cd "${1:-${DEFAULT_REFS_DIR}}" && pwd)"
TIMESTAMP="${2:-}"

mkdir -p "${HOST_CACHE_DIR}"

if [[ "${BUILD_IMAGE}" == "1" ]] || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

docker_args=(
  --rm
  --user
  "${HOST_UID}:${HOST_GID}"
  --gpus
  all
  -e
  HOME=/home/app
  -e
  XDG_CACHE_HOME=/home/app/.cache
  -e
  HF_HOME=/home/app/.cache/huggingface
  -e
  TORCH_HOME=/home/app/.cache/torch
  -v
  "${APP_DIR}:/app"
  -v
  "${HOST_CACHE_DIR}:/home/app/.cache"
  -v
  "${REFS_DIR}:/refs:ro"
  "${IMAGE_NAME}"
  python
  /app/model-runners/chatterbox_turbo/run_inner.py
  --refs
  /refs
  --output
  "${OUTPUT_DIR}"
  --device
  "${DEVICE}"
  --temperature
  "${TEMPERATURE}"
  --top-p
  "${TOP_P}"
  --top-k
  "${TOP_K}"
  --repetition-penalty
  "${REPETITION_PENALTY}"
)

if [[ -n "${TIMESTAMP}" ]]; then
  docker_args+=(--timestamp "${TIMESTAMP}")
fi

docker run "${docker_args[@]}"
