#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TICKER_AGENTS_STACK="${TICKER_AGENTS_STACK:-fmp}"
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/run_ticker_agents.py "$@"
