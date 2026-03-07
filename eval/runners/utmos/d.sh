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
BATCH_SIZE="${UTMOS_BATCH_SIZE:-32}"
SHM_SIZE="${UTMOS_SHM_SIZE:-8g}"
REMOVE_SILENT_SECTION="${UTMOS_REMOVE_SILENT_SECTION:-false}"
NUM_WORKERS="${UTMOS_NUM_WORKERS:-2}"
CPU_THREADS="${UTMOS_CPU_THREADS:-1}"
SPEC_MIXUP_INNER="${UTMOS_SPEC_MIXUP_INNER:-}"
SPEC_NUM_FRAMES="${UTMOS_SPEC_NUM_FRAMES:-}"

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
  -e
  OMP_NUM_THREADS="${CPU_THREADS}"
  -e
  MKL_NUM_THREADS="${CPU_THREADS}"
  -e
  OPENBLAS_NUM_THREADS="${CPU_THREADS}"
  -e
  NUMEXPR_NUM_THREADS="${CPU_THREADS}"
  -e
  VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
  -e
  BLIS_NUM_THREADS="${CPU_THREADS}"
  -e
  NUMBA_NUM_THREADS="${CPU_THREADS}"
  -e
  TORCH_NUM_THREADS="${CPU_THREADS}"
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
if [[ -n "${SPEC_MIXUP_INNER}" ]]; then
  docker_args+=(--spec-mixup-inner "${SPEC_MIXUP_INNER}")
fi
if [[ -n "${SPEC_NUM_FRAMES}" ]]; then
  docker_args+=(--spec-num-frames "${SPEC_NUM_FRAMES}")
fi

docker run "${docker_args[@]}"
