from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import date

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


FINAL_BEGIN = "=== FINAL_DECISION_MARKDOWN_BEGIN ==="
FINAL_END = "=== FINAL_DECISION_MARKDOWN_END ==="
DEFAULT_TICKER = "SPY"


def resolve_inputs(ticker: str | None = None, analysis_date: str | None = None) -> tuple[str, str]:
    ticker = (ticker or os.getenv("TICKER") or DEFAULT_TICKER).strip().upper()
    analysis_date = (analysis_date or os.getenv("ANALYSIS_DATE") or date.today().isoformat()).strip()
    return ticker, analysis_date


def build_config():
    # Reuse the existing LAN services instead of local API keys:
    # - local OpenAI-compatible Qwen on 10.17.17.99
    # - Massive/Perplexity/Grok via MCP gateways on 10.17.17.90
    os.environ.setdefault("OPENAI_API_KEY", "dummy")

    config = deepcopy(DEFAULT_CONFIG)
    config["llm_provider"] = "openai"
    config["backend_url"] = "http://10.17.17.99:8005/v1"
    config["deep_think_llm"] = "Qwen/Qwen3.6-27B-FP8"
    config["quick_think_llm"] = "Qwen/Qwen3.6-27B-FP8"
    config["market_data_mcp_url"] = "https://10.17.17.90:8083/mcp"
    config["news_mcp_url"] = "http://10.17.17.90:9081/mcp"
    config["mcp_verify_tls"] = False

    # Vendor routing added in this patch set.
    config["data_vendors"]["core_stock_apis"] = "massive"
    config["data_vendors"]["fundamental_data"] = "massive"
    config["data_vendors"]["news_data"] = "perplexity"
    config["tool_vendors"]["get_news"] = "perplexity"
    config["tool_vendors"]["get_global_news"] = "grok"
    config["sentiment_x_source"] = "grok"

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grounded ticker-agent stack.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol, defaults to SPY")
    parser.add_argument("analysis_date", nargs="?", help="Analysis date YYYY-MM-DD, defaults to today")
    return parser.parse_args()


def main():
    args = parse_args()
    ticker, analysis_date = resolve_inputs(args.ticker, args.analysis_date)
    config = build_config()
    graph = TradingAgentsGraph(debug=True, config=config)
    _, decision = graph.propagate(ticker, analysis_date)
    print(FINAL_BEGIN)
    print(decision)
    print(FINAL_END)


if __name__ == "__main__":
    main()
