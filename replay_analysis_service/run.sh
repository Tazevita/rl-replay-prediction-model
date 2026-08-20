#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="$SCRIPT_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
exec "$SCRIPT_DIR/../prediction-model/.venv/bin/python" -m replay_analysis_service "$@"
