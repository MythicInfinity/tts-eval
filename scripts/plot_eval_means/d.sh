#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-tts-eval-plot-eval-means}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_IMAGE="${BUILD_IMAGE:-0}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_CACHE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}"

mkdir -p "${HOST_CACHE_DIR}"

if [[ "${BUILD_IMAGE}" == "1" ]] || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  docker build \
    -f "${SCRIPT_DIR}/Dockerfile" \
    -t "${IMAGE_NAME}" \
    "${REPO_ROOT}"
fi

docker run --rm \
  --user "${HOST_UID}:${HOST_GID}" \
  -e HOME=/home/app \
  -e XDG_CACHE_HOME=/home/app/.cache \
  -e MPLCONFIGDIR=/home/app/.cache/matplotlib \
  -v "${REPO_ROOT}:/app" \
  -v "${HOST_CACHE_DIR}:/home/app/.cache" \
  -w /app \
  "${IMAGE_NAME}" \
  python /app/scripts/plot_eval_means/run_inner.py "$@"
