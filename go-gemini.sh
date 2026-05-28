#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TICKER_AGENTS_STACK="fmp"
export TRADINGAGENTS_LLM_PROVIDER="google"
export TRADINGAGENTS_DEEP_THINK_LLM="gemini-3.5-flash"
export TRADINGAGENTS_QUICK_THINK_LLM="gemini-3.5-flash"
export TRADINGAGENTS_LLM_BACKEND_URL=""

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/run_ticker_agents.py "$@"
