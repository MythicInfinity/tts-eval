#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_INPUTS_DIR="${REPO_ROOT}/data/inputs"
DEFAULT_REFS_DIR="${REPO_ROOT}/data/refs"

if [[ $# -gt 3 ]]; then
  echo "usage: run_all.sh [inputs_dir] [refs_dir] [timestamp]" >&2
  exit 1
fi

INPUTS_DIR="${1:-${DEFAULT_INPUTS_DIR}}"
REFS_DIR="${2:-${DEFAULT_REFS_DIR}}"
TIMESTAMP="${3:-}"

cd "${REPO_ROOT}"
"${REPO_ROOT}/eval/runners/ctc/d.sh" "${INPUTS_DIR}" "${REFS_DIR}" "${TIMESTAMP}"
"${REPO_ROOT}/eval/runners/ttsds2/d.sh" "${INPUTS_DIR}" "${REFS_DIR}" "${TIMESTAMP}"
"${REPO_ROOT}/eval/runners/dnsmos/d.sh" "${INPUTS_DIR}" "${REFS_DIR}" "${TIMESTAMP}"
"${REPO_ROOT}/eval/runners/nisqa/d.sh" "${INPUTS_DIR}" "${REFS_DIR}" "${TIMESTAMP}"
