#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-tts-eval-utmos}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
APP_DIR="${REPO_ROOT}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_CACHE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}"
DEFAULT_INPUTS_DIR="${REPO_ROOT}/data/inputs"
DEFAULT_REFS_DIR="${REPO_ROOT}/data/refs"
DEVICE="${UTMOS_DEVICE:-cuda:0}"
BATCH_SIZE="${UTMOS_BATCH_SIZE:-16}"
SHM_SIZE="${UTMOS_SHM_SIZE:-2g}"
REMOVE_SILENT_SECTION="${UTMOS_REMOVE_SILENT_SECTION:-true}"
DEFAULT_NUM_WORKERS="$(nproc 2>/dev/null || echo 1)"
if (( DEFAULT_NUM_WORKERS > 8 )); then
  DEFAULT_NUM_WORKERS=8
fi
if (( DEFAULT_NUM_WORKERS < 1 )); then
  DEFAULT_NUM_WORKERS=1
fi
NUM_WORKERS="${UTMOS_NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"

if [[ $# -gt 3 ]]; then
  echo "usage: d.sh [inputs_dir] [refs_dir] [timestamp]" >&2
  exit 1
fi

INPUTS_DIR="$(cd "${1:-${DEFAULT_INPUTS_DIR}}" && pwd)"
REFS_DIR="$(cd "${2:-${DEFAULT_REFS_DIR}}" && pwd)"
TIMESTAMP="${3:-}"
OUTPUT_DIR="/app/data/evals/utmos"

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
  --shm-size
  "${SHM_SIZE}"
  -e
  HOME=/home/app
  -e
  XDG_CACHE_HOME=/home/app/.cache
  -e
  HF_HOME=/home/app/.cache/huggingface
  -e
  TORCH_HOME=/home/app/.cache/torch
  -e
  UTMOSV2_CHACHE=/home/app/.cache/utmosv2
  -v
  "${APP_DIR}:/app"
  -v
  "${HOST_CACHE_DIR}:/home/app/.cache"
  -v
  "${INPUTS_DIR}:/inputs:ro"
  -v
  "${REFS_DIR}:/refs:ro"
  "${IMAGE_NAME}"
  python
  /app/eval/runners/utmos/run_inner.py
  --inputs
  /inputs
  --refs
  /refs
  --output
  "${OUTPUT_DIR}"
  --batch-size
  "${BATCH_SIZE}"
  --num-workers
  "${NUM_WORKERS}"
  --remove-silent-section
  "${REMOVE_SILENT_SECTION}"
  --device
  "${DEVICE}"
)

if [[ -n "${TIMESTAMP}" ]]; then
  docker_args+=(--timestamp "${TIMESTAMP}")
fi

docker run "${docker_args[@]}"
