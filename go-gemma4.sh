#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENDPOINT="${GEMMA4_ENDPOINT:-10.17.17.99:8000}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export TICKER_AGENTS_STACK="${TICKER_AGENTS_STACK:-fmp}"
export TRADINGAGENTS_LLM_PROVIDER="openai"
export TRADINGAGENTS_DEEP_THINK_LLM="google/gemma-4-31B-it"
export TRADINGAGENTS_QUICK_THINK_LLM="google/gemma-4-31B-it"
export TRADINGAGENTS_LLM_BACKEND_URL="http://$ENDPOINT/v1"

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python scripts/run_ticker_agents.py "$@"
