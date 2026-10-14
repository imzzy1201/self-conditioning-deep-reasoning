#!/usr/bin/env bash
# common/env.sh — resolve the repository root and shared paths.
#
# Every run script sources this file. The root is derived from this file's own
# location (not the caller's $PWD), so any stage is callable from anywhere.
# All relative data/output overrides in config.yaml are resolved against the
# root computed here.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/../common/env.sh"

if [[ -z "${_SCDR_ENV_LOADED:-}" ]]; then
  _SCDR_ENV_LOADED=1

  COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # .../scripts/common
  REPO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"             # .../  (repo root)

  export REPO_ROOT
  export COMMON_DIR

  # Default runtime roots (overridable via env for isolated runs).
  export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
  export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
  export REFERENCE_ROOT="${REFERENCE_ROOT:-${REPO_ROOT}/results/reference}"

  # Load .env if present (API keys / model ids). Never fails if absent.
  if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
  fi
fi

# Shared helpers --------------------------------------------------------------

# resolve_path <value> [base] — return an absolute path, falling back to $base
# (default: $REPO_ROOT) for relative values.
resolve_path() {
  local value="$1"
  local base="${2:-${REPO_ROOT}}"
  if [[ -z "${value}" ]]; then
    printf '%s' "${base}"
  elif [[ "${value:0:1}" == "/" ]]; then
    printf '%s' "${value}"
  else
    printf '%s' "${base}/${value}"
  fi
}