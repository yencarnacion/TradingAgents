from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import date

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


FINAL_BEGIN = "=== FINAL_DECISION_MARKDOWN_BEGIN ==="
FINAL_END = "=== FINAL_DECISION_MARKDOWN_END ==="
STATE_BEGIN = "=== FINAL_STATE_REPORTS_JSON_BEGIN ==="
STATE_END = "=== FINAL_STATE_REPORTS_JSON_END ==="
DEFAULT_TICKER = "SPY"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.6-27B-FP8"
DEFAULT_QWEN_BACKEND_URL = "http://10.17.17.99:8005/v1"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def _env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def report_payload(final_state: dict) -> dict:
    return {
        "market_report": final_state.get("market_report", ""),
        "sentiment_report": final_state.get("sentiment_report", ""),
        "news_report": final_state.get("news_report", ""),
        "fundamentals_report": final_state.get("fundamentals_report", ""),
        "investment_debate_state": final_state.get("investment_debate_state", {}),
        "investment_plan": final_state.get("investment_plan", ""),
        "trader_investment_decision": final_state.get("trader_investment_plan", ""),
        "risk_debate_state": final_state.get("risk_debate_state", {}),
    }


def resolve_inputs(ticker: str | None = None, analysis_date: str | None = None) -> tuple[str, str]:
    ticker = (ticker or os.getenv("TICKER") or DEFAULT_TICKER).strip().upper()
    analysis_date = (analysis_date or os.getenv("ANALYSIS_DATE") or date.today().isoformat()).strip()
    return ticker, analysis_date


def build_config():
    # Reuse the existing LAN services instead of local API keys:
    # - local OpenAI-compatible Qwen on 10.17.17.99
    # - Massive MCP on 10.17.17.90:8083 for intraday / options / tape data
    # - FMP MCP on 10.17.17.90:8086 for fundamentals / news / insiders / macro context
    # - news/Grok MCP on 10.17.17.90:9081 for X sentiment fallback when no local xAI key is present
    config = deepcopy(DEFAULT_CONFIG)

    provider = _env_value("TRADINGAGENTS_LLM_PROVIDER") or "openai"
    default_model = DEFAULT_GEMINI_MODEL if provider == "google" else DEFAULT_QWEN_MODEL
    default_backend_url = DEFAULT_QWEN_BACKEND_URL if provider == "openai" else None

    if provider == "openai":
        os.environ.setdefault("OPENAI_API_KEY", "dummy")

    config["llm_provider"] = provider
    config["backend_url"] = _env_value("TRADINGAGENTS_LLM_BACKEND_URL") or default_backend_url
    config["deep_think_llm"] = _env_value("TRADINGAGENTS_DEEP_THINK_LLM") or default_model
    config["quick_think_llm"] = _env_value("TRADINGAGENTS_QUICK_THINK_LLM") or default_model
    config["market_data_mcp_url"] = "https://10.17.17.90:8083/mcp"
    config["fmp_mcp_url"] = "http://10.17.17.90:8086/mcp"
    config["news_mcp_url"] = "http://10.17.17.90:9081/mcp"
    config["mcp_verify_tls"] = False

    config["data_vendors"]["core_stock_apis"] = "massive"
    config["data_vendors"]["fundamental_data"] = "fmp"
    config["data_vendors"]["news_data"] = "fmp"
    config["tool_vendors"]["get_stock_data"] = "massive"
    config["tool_vendors"]["get_fundamentals"] = "fmp"
    config["tool_vendors"]["get_balance_sheet"] = "fmp"
    config["tool_vendors"]["get_cashflow"] = "fmp"
    config["tool_vendors"]["get_income_statement"] = "fmp"
    config["tool_vendors"]["get_news"] = "fmp,grok"
    config["tool_vendors"]["get_global_news"] = "grok,fmp"
    config["tool_vendors"]["get_insider_transactions"] = "fmp"
    # Technical indicators can be calculated locally from Massive daily OHLCV,
    # so they no longer need the yfinance stockstats fallback.

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TradingAgents with direct FMP MCP-backed dataflows.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol, defaults to SPY")
    parser.add_argument("analysis_date", nargs="?", help="Analysis date YYYY-MM-DD, defaults to today")
    return parser.parse_args()


def main():
    args = parse_args()
    ticker, analysis_date = resolve_inputs(args.ticker, args.analysis_date)
    config = build_config()
    graph = TradingAgentsGraph(debug=True, config=config)
    final_state, decision = graph.propagate(ticker, analysis_date)
    print(STATE_BEGIN)
    print(json.dumps(report_payload(final_state), ensure_ascii=False, default=str))
    print(STATE_END)
    print(FINAL_BEGIN)
    print(decision)
    print(FINAL_END)


if __name__ == "__main__":
    main()
