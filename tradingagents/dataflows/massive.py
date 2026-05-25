from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import requests

from .config import get_config
from .mcp_support import MCPToolError, call_tool, market_data_mcp_url

_MASSIVE_BASE_URL = os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")


def _massive_api_key() -> Optional[str]:
    return os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")


def _auth_headers() -> Dict[str, str]:
    api_key = _massive_api_key()
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _request(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    api_key = _massive_api_key()
    if api_key:
        response = requests.get(
            f"{_MASSIVE_BASE_URL}{path}",
            headers=_auth_headers(),
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("status") not in (None, "OK"):
            raise RuntimeError(payload.get("error") or payload.get("message") or str(payload))
        return payload

    mcp_url = market_data_mcp_url()
    if not mcp_url:
        raise RuntimeError("MASSIVE_API_KEY/POLYGON_API_KEY is not set and market_data_mcp_url is not configured")

    payload = call_tool(mcp_url, "call_api", {"method": "GET", "path": path, "params": params or {}})
    raw = (payload or {}).get("result", "")
    if not raw:
        return {"results": []}

    raw = raw.strip()
    if raw.startswith("Error"):
        raise MCPToolError(raw)
    if raw.startswith("{") or raw.startswith("["):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"results": parsed}

    reader = csv.DictReader(io.StringIO(raw))
    return {"results": list(reader)}


def _header(title: str) -> str:
    return f"# {title}\n# Data source: Massive\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"


def _rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = payload.get("results") or []
    if isinstance(results, dict):
        return [results]
    return list(results)


def _to_csv_block(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<no rows returned>"
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def _format_articles(title: str, articles: Iterable[Dict[str, Any]]) -> str:
    lines = [_header(title).rstrip(), ""]
    count = 0
    for article in articles:
        lines.append(f"### {article.get('title', 'Untitled')} (source: {article.get('publisher_name', 'Unknown')})")
        if article.get("published_utc"):
            lines.append(f"Published: {article['published_utc']}")
        if article.get("description"):
            lines.append(article["description"])
        for insight in article.get("insights", []) or []:
            ticker = insight.get("ticker", "unknown")
            sentiment = insight.get("sentiment", "unknown")
            reasoning = insight.get("sentiment_reasoning", "")
            lines.append(f"Insight ({ticker} / {sentiment}): {reasoning}")
        if article.get("article_url"):
            lines.append(f"Link: {article['article_url']}")
        lines.append("")
        count += 1
    if count == 0:
        return f"{_header(title)}No news returned."
    return "\n".join(lines).strip() + "\n"


def get_stock_data(symbol: str, start_date: str, end_date: str):
    try:
        payload = _request(
            f"/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start_date}/{end_date}",
            params={"adjusted": "true", "sort": "asc", "limit": 50000},
        )
        rows = _rows(payload)
        if not rows:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        normalized = []
        for row in rows:
            timestamp_ms = row.get("t")
            trade_date = None
            if timestamp_ms not in (None, ""):
                timestamp_ms = int(float(timestamp_ms))
                trade_date = datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d")
            normalized.append(
                {
                    "date": trade_date,
                    "open": row.get("o"),
                    "high": row.get("h"),
                    "low": row.get("l"),
                    "close": row.get("c"),
                    "volume": row.get("v"),
                    "vwap": row.get("vw"),
                    "transactions": row.get("n"),
                    "timestamp_ms": timestamp_ms,
                }
            )
        return _header(f"Stock data for {symbol.upper()} from {start_date} to {end_date}") + _to_csv_block(normalized)
    except Exception as e:
        return f"Error retrieving Massive stock data for {symbol}: {e}"


def get_fundamentals(ticker: str, curr_date: Optional[str] = None):
    try:
        overview_payload = _request(f"/v3/reference/tickers/{ticker.upper()}", params={"date": curr_date} if curr_date else None)
        overview_rows = _rows(overview_payload)
        overview = overview_rows[0] if overview_rows else (overview_payload.get("results", {}) or {})
        ratios_payload = _request("/stocks/financials/v1/ratios", params={"ticker": ticker.upper(), "limit": 1})
        ratios_list = _rows(ratios_payload)
        ratios = ratios_list[0] if ratios_list else {}

        fields = [
            ("Name", overview.get("name")),
            ("Description", overview.get("description")),
            ("Primary Exchange", overview.get("primary_exchange")),
            ("Industry", overview.get("sic_description")),
            ("Employees", overview.get("total_employees")),
            ("Market Cap", overview.get("market_cap")),
            ("CIK", overview.get("cik")),
            ("Price", ratios.get("price")),
            ("Average Volume", ratios.get("average_volume")),
            ("P/E", ratios.get("price_to_earnings")),
            ("P/B", ratios.get("price_to_book")),
            ("P/S", ratios.get("price_to_sales")),
            ("EV/Sales", ratios.get("ev_to_sales")),
            ("EV/EBITDA", ratios.get("ev_to_ebitda")),
            ("EPS", ratios.get("earnings_per_share")),
            ("Dividend Yield", ratios.get("dividend_yield")),
            ("ROA", ratios.get("return_on_assets")),
            ("ROE", ratios.get("return_on_equity")),
            ("Debt to Equity", ratios.get("debt_to_equity")),
            ("Current Ratio", ratios.get("current")),
            ("Quick Ratio", ratios.get("quick")),
            ("Cash Ratio", ratios.get("cash")),
            ("Enterprise Value", ratios.get("enterprise_value")),
            ("Latest Ratio Period End", ratios.get("period_end")),
        ]
        lines = [f"{label}: {_clean_value(value)}" for label, value in fields if value is not None]
        if not lines:
            return f"No fundamentals data found for symbol '{ticker}'"
        return _header(f"Company Fundamentals for {ticker.upper()}") + "\n".join(lines)
    except Exception as e:
        return f"Error retrieving Massive fundamentals for {ticker}: {e}"


def _financial_statement(path: str, label: str, ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None) -> str:
    params: Dict[str, Any] = {
        "tickers": ticker.upper(),
        "timeframe": freq.lower(),
        "limit": 8,
        "sort": "period_end.desc",
    }
    if curr_date:
        params["period_end.lte"] = curr_date

    payload = _request(path, params=params)
    rows = _rows(payload)
    if not rows:
        return f"No {label.lower()} data found for symbol '{ticker}'"
    return _header(f"{label} data for {ticker.upper()} ({freq})") + _to_csv_block(rows)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None):
    try:
        return _financial_statement("/stocks/financials/v1/balance-sheets", "Balance Sheet", ticker, freq, curr_date)
    except Exception as e:
        return f"Error retrieving Massive balance sheet for {ticker}: {e}"


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None):
    try:
        return _financial_statement("/stocks/financials/v1/cash-flow-statements", "Cash Flow", ticker, freq, curr_date)
    except Exception as e:
        return f"Error retrieving Massive cash flow for {ticker}: {e}"


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None):
    try:
        return _financial_statement("/stocks/financials/v1/income-statements", "Income Statement", ticker, freq, curr_date)
    except Exception as e:
        return f"Error retrieving Massive income statement for {ticker}: {e}"


def get_news(ticker: str, start_date: str, end_date: str):
    try:
        article_limit = get_config()["news_article_limit"]
        payload = _request(
            "/v2/reference/news",
            params={
                "ticker": ticker.upper(),
                "published_utc.gte": f"{start_date}T00:00:00Z",
                "published_utc.lte": f"{end_date}T23:59:59Z",
                "sort": "published_utc",
                "order": "desc",
                "limit": article_limit,
            },
        )
        articles = []
        for item in _rows(payload):
            publisher = item.get("publisher") or {}
            articles.append(
                {
                    "title": item.get("title"),
                    "publisher_name": publisher.get("name"),
                    "published_utc": item.get("published_utc"),
                    "description": item.get("description"),
                    "article_url": item.get("article_url"),
                    "insights": item.get("insights") or [],
                }
            )
        if not articles:
            return f"No news found for {ticker} between {start_date} and {end_date}"
        return _format_articles(f"{ticker.upper()} News, from {start_date} to {end_date}", articles)
    except Exception as e:
        return f"Error retrieving Massive news for {ticker}: {e}"
