#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUTS_DIR="${1:?usage: run_all.sh <inputs_dir> <refs_dir> [timestamp] }"
REFS_DIR="${2:?usage: run_all.sh <inputs_dir> <refs_dir> [timestamp] }"
TIMESTAMP="${3:-}"

cd "${REPO_ROOT}"
"${REPO_ROOT}/eval/runners/ctc/d.sh" "${INPUTS_DIR}" "${REFS_DIR}" "${TIMESTAMP}"
