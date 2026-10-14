#!/usr/bin/env bash
# common/flops.sh — manual convenience wrapper around src/experiment_tools/flops_cli.py.
#
# No stage script calls this wrapper: 05_match_checkpoints.sh invokes flops_cli.py
# directly. Real measurement happens in each leaf's 02_measure_flops.sh (trainer
# under --measure_flops). The measure subcommand prints the measurement contract.
# The pairing rules live only in src/experiment_tools/flops.py, per docs/flops_protocol.md.
#
#   flops.sh <measure|apply|audit|match|inference> --config <experiment config.yaml> [extra]
#   (exception: inference takes --cells <dir>, not --config)

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SELF_DIR}/env.sh"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/src/experiment_tools:${PYTHONPATH:-}"

exec python "${REPO_ROOT}/src/experiment_tools/flops_cli.py" "$@"