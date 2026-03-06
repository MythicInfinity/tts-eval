#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-tts-eval-ctc}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
APP_DIR="${SCRIPT_DIR}"

INPUTS_DIR="$(cd "${1:?usage: d.sh <inputs_dir> <refs_dir> [timestamp] }" && pwd)"
REFS_DIR="$(cd "${2:?usage: d.sh <inputs_dir> <refs_dir> [timestamp] }" && pwd)"
TIMESTAMP="${3:-}"
OUTPUT_DIR="/app/data/evals/ctc"

if [[ "${BUILD_IMAGE}" == "1" ]] || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

docker_args=(
  --rm
  --gpus
  all
  -v
  "${APP_DIR}:/app"
  -v
  "${REPO_ROOT}/src:/app/src:ro"
  -v
  "${INPUTS_DIR}:/inputs:ro"
  -v
  "${REFS_DIR}:/refs:ro"
  "${IMAGE_NAME}"
  python
  /app/run_inner.py
  --inputs
  /inputs
  --refs
  /refs
  --output
  "${OUTPUT_DIR}"
  --device cuda
)

if [[ -n "${TIMESTAMP}" ]]; then
  docker_args+=(--timestamp "${TIMESTAMP}")
fi

docker run "${docker_args[@]}"
