from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import get_config
from .mcp_support import MCPToolError, call_tool


_INDICATOR_MAP: Dict[str, tuple[str, int, str]] = {
    "close_50_sma": ("simple-moving-average", 50, "sma"),
    "close_200_sma": ("simple-moving-average", 200, "sma"),
    "close_10_ema": ("exponential-moving-average", 10, "ema"),
    "rsi": ("relative-strength-index", 14, "rsi"),
    "williams": ("williams", 14, "williams"),
}


def _fmp_mcp_url() -> Optional[str]:
    return get_config().get("fmp_mcp_url")


def _header(title: str) -> str:
    return f"# {title}\n# Data source: FMP MCP\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"


def _call(tool_name: str, arguments: Dict[str, Any]) -> Any:
    mcp_url = _fmp_mcp_url()
    if not mcp_url:
        raise RuntimeError("fmp_mcp_url is not configured")

    payload = call_tool(mcp_url, tool_name, arguments)
    structured = payload or {}
    raw = structured.get("result", "") if isinstance(structured, dict) else structured
    if raw in (None, ""):
        return []
    if isinstance(raw, (list, dict)):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("Error"):
            raise MCPToolError(raw)
        if raw.startswith("{") or raw.startswith("["):
            return json.loads(raw)
        return raw
    return raw


def _rows(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _sorted_rows(rows: Sequence[Dict[str, Any]], key: str = "date", reverse: bool = False) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row.get(key) or ""), reverse=reverse)


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


def _first_row(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    rows = _rows(_call(tool_name, arguments))
    return rows[0] if rows else {}


def _format_articles(title: str, articles: Iterable[Dict[str, Any]]) -> str:
    lines = [_header(title).rstrip(), ""]
    count = 0
    for article in articles:
        lines.append(f"### {article.get('title', 'Untitled')} (source: {article.get('publisher') or article.get('site') or 'Unknown'})")
        if article.get("publishedDate"):
            lines.append(f"Published: {article['publishedDate']}")
        if article.get("text"):
            lines.append(article["text"])
        if article.get("symbol"):
            lines.append(f"Primary symbol: {article['symbol']}")
        if article.get("url"):
            lines.append(f"Link: {article['url']}")
        lines.append("")
        count += 1
    if count == 0:
        return f"{_header(title)}No news returned."
    return "\n".join(lines).strip() + "\n"


def _statement_period(freq: str) -> str:
    normalized = (freq or "quarterly").strip().lower()
    if normalized in {"quarter", "quarterly", "q", "qtr"}:
        return "quarter"
    if normalized in {"annual", "year", "yearly", "fy"}:
        return "annual"
    return normalized


def _parse_iso_date(value: Any) -> Optional[datetime.date]:
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text[:10], text):
        try:
            return datetime.strptime(candidate, "%Y-%m-%d").date()
        except ValueError:
            continue
    return None


def _earnings_anchor_applicable(profile: Dict[str, Any]) -> bool:
    if not profile:
        return True
    for key in ("isEtf", "isFund", "fund", "etf"):
        value = profile.get(key)
        if isinstance(value, bool) and value:
            return False
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "y"}:
            return False
    return True


def get_stock_data(symbol: str, start_date: str, end_date: str):
    try:
        rows = _rows(
            _call(
                "chart",
                {
                    "endpoint": "historical-price-eod-full",
                    "symbol": symbol.upper(),
                    "from_date": start_date,
                    "to_date": end_date,
                },
            )
        )
        rows = _sorted_rows(rows)
        if not rows:
            return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        normalized = [
            {
                "date": row.get("date"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
                "change": row.get("change"),
                "changePercent": row.get("changePercent"),
                "vwap": row.get("vwap"),
            }
            for row in rows
        ]
        return _header(f"Stock data for {symbol.upper()} from {start_date} to {end_date}") + _to_csv_block(normalized)
    except Exception as e:
        return f"Error retrieving FMP stock data for {symbol}: {e}"


def get_fundamentals(ticker: str, curr_date: Optional[str] = None):
    try:
        profile = _first_row("company", {"endpoint": "profile-symbol", "symbol": ticker.upper()})
        quote = _first_row("quote", {"endpoint": "quote", "symbol": ticker.upper()})
        metrics = _first_row("statements", {"endpoint": "metrics-ratios-ttm", "symbol": ticker.upper()})
        scores = _first_row("statements", {"endpoint": "financial-scores", "symbol": ticker.upper()})

        fields = [
            ("Name", profile.get("companyName") or profile.get("name")),
            ("Description", profile.get("description")),
            ("Exchange", profile.get("exchangeShortName") or profile.get("exchange")),
            ("Sector", profile.get("sector")),
            ("Industry", profile.get("industry")),
            ("Country", profile.get("country")),
            ("CEO", profile.get("ceo")),
            ("Employees", profile.get("fullTimeEmployees")),
            ("Market Cap", quote.get("marketCap") or profile.get("mktCap") or scores.get("marketCap")),
            ("Price", quote.get("price")),
            ("Beta", profile.get("beta")),
            ("52 Week Range", f"{profile.get('range')}" if profile.get("range") else None),
            ("Average Volume", profile.get("averageVolume") or quote.get("volume")),
            ("Dividend Yield", metrics.get("dividendYieldTTM")),
            ("P/E", profile.get("pe") or metrics.get("priceToEarningsRatioTTM")),
            ("P/B", profile.get("pb") or metrics.get("priceToBookRatioTTM") or metrics.get("priceToBookTTM")),
            ("P/S", profile.get("priceToSalesRatio") or metrics.get("priceToSalesRatioTTM")),
            ("ROA", metrics.get("returnOnAssetsTTM")),
            ("ROE", metrics.get("returnOnEquityTTM")),
            ("Current Ratio", metrics.get("currentRatioTTM")),
            ("Debt to Equity", metrics.get("debtToEquityRatioTTM")),
            ("Net Profit Margin", metrics.get("netProfitMarginTTM")),
            ("Enterprise Value", metrics.get("enterpriseValueTTM")),
            ("Altman Z Score", scores.get("altmanZScore")),
            ("Piotroski Score", scores.get("piotroskiScore")),
        ]
        lines = [f"{label}: {_clean_value(value)}" for label, value in fields if value is not None]
        if curr_date:
            lines.append(f"Requested as-of date: {curr_date}")
        if not lines:
            return f"No fundamentals data found for symbol '{ticker}'"
        return _header(f"Company Fundamentals for {ticker.upper()}") + "\n".join(lines)
    except Exception as e:
        return f"Error retrieving FMP fundamentals for {ticker}: {e}"


def _financial_statement(endpoint: str, label: str, ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None):
    rows = _rows(
        _call(
            "statements",
            {
                "endpoint": endpoint,
                "symbol": ticker.upper(),
                "period": _statement_period(freq),
                "limit": 8,
            },
        )
    )
    rows = _sorted_rows(rows, reverse=True)
    if curr_date:
        rows = [row for row in rows if str(row.get("date") or "") <= curr_date]
    if not rows:
        return f"No {label.lower()} data found for symbol '{ticker}'"
    return _header(f"{label} data for {ticker.upper()} ({freq})") + _to_csv_block(rows)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None):
    try:
        return _financial_statement("balance-sheet-statement", "Balance Sheet", ticker, freq, curr_date)
    except Exception as e:
        return f"Error retrieving FMP balance sheet for {ticker}: {e}"


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None):
    try:
        return _financial_statement("cashflow-statement", "Cash Flow", ticker, freq, curr_date)
    except Exception as e:
        return f"Error retrieving FMP cash flow for {ticker}: {e}"


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: Optional[str] = None):
    try:
        return _financial_statement("income-statement", "Income Statement", ticker, freq, curr_date)
    except Exception as e:
        return f"Error retrieving FMP income statement for {ticker}: {e}"


def get_recent_earnings_anchor_data(ticker: str, curr_date: str) -> Dict[str, Any]:
    profile = _first_row("company", {"endpoint": "profile-symbol", "symbol": ticker.upper()})
    if not _earnings_anchor_applicable(profile):
        return {}

    target_date = datetime.strptime(curr_date, "%Y-%m-%d").date()
    rows = _rows(
        _call(
            "calendar",
            {
                "endpoint": "earnings-company",
                "symbol": ticker.upper(),
            },
        )
    )
    eligible: List[tuple[datetime.date, Dict[str, Any]]] = []
    for row in rows:
        report_date = _parse_iso_date(row.get("date") or row.get("reportedDate") or row.get("fiscalDateEnding"))
        if report_date and report_date <= target_date:
            eligible.append((report_date, row))

    if not eligible:
        return {}

    report_date, row = max(eligible, key=lambda item: item[0])
    time_label = str(row.get("time") or "").strip().upper()
    timing = time_label if time_label in {"BMO", "AMC", "DMT"} else ""
    anchor_label = f"Most recent earnings ({timing + ' ' if timing else ''}{report_date.isoformat()})"
    return {
        "ticker": ticker.upper(),
        "anchor_date": report_date.isoformat(),
        "anchor_label": anchor_label,
        "time": time_label or None,
        "eps": _clean_value(row.get("eps") or row.get("epsActual") or row.get("epsReported")),
        "epsEstimated": _clean_value(row.get("epsEstimated") or row.get("estimatedEps")),
        "revenue": _clean_value(row.get("revenue") or row.get("revenueActual")),
        "revenueEstimated": _clean_value(row.get("revenueEstimated") or row.get("estimatedRevenue")),
        "source_vendor": "fmp",
    }


def get_recent_earnings_anchor(ticker: str, curr_date: str):
    try:
        profile = _first_row("company", {"endpoint": "profile-symbol", "symbol": ticker.upper()})
        if not _earnings_anchor_applicable(profile):
            return (
                f"Earnings anchor not applicable for {ticker.upper()}: "
                "instrument appears to be an ETF or fund rather than an operating company."
            )
        anchor = get_recent_earnings_anchor_data(ticker, curr_date)
        if not anchor:
            return f"No earnings anchor found for symbol '{ticker}' on or before {curr_date}"
        rows = [
            {
                "ticker": anchor.get("ticker"),
                "anchor_date": anchor.get("anchor_date"),
                "time": anchor.get("time"),
                "anchor_label": anchor.get("anchor_label"),
                "eps": anchor.get("eps"),
                "epsEstimated": anchor.get("epsEstimated"),
                "revenue": anchor.get("revenue"),
                "revenueEstimated": anchor.get("revenueEstimated"),
                "note": "Use this earnings date as the anchored VWAP (AVWAP) start point.",
            }
        ]
        return _header(f"Recent Earnings Anchor for {ticker.upper()}") + _to_csv_block(rows)
    except Exception as e:
        return f"Error retrieving FMP earnings anchor for {ticker}: {e}"


def get_news(ticker: str, start_date: str, end_date: str):
    try:
        article_limit = get_config()["news_article_limit"]
        rows = _rows(
            _call(
                "news",
                {
                    "endpoint": "stock-news",
                    "symbols": [ticker.upper()],
                    "from_date": start_date,
                    "to_date": end_date,
                    "limit": article_limit,
                },
            )
        )
        filtered = [row for row in rows if str(row.get("symbol") or "").upper() == ticker.upper()]
        articles = filtered or rows
        if not articles:
            return f"No news found for {ticker} between {start_date} and {end_date}"
        return _format_articles(f"{ticker.upper()} News, from {start_date} to {end_date}", articles[:article_limit])
    except Exception as e:
        return f"Error retrieving FMP news for {ticker}: {e}"


def get_global_news(curr_date: str, look_back_days: Optional[int] = None, limit: Optional[int] = None):
    try:
        config = get_config()
        look_back_days = look_back_days or config["global_news_lookback_days"]
        limit = limit or config["global_news_article_limit"]
        start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
        rows = _rows(
            _call(
                "news",
                {
                    "endpoint": "general-news",
                    "from_date": start_dt.strftime("%Y-%m-%d"),
                    "to_date": curr_date,
                    "limit": limit,
                },
            )
        )
        if not rows:
            return f"No global news found for {curr_date}"
        return _format_articles(
            f"Global Market News, from {start_dt.strftime('%Y-%m-%d')} to {curr_date}",
            rows[:limit],
        )
    except Exception as e:
        return f"Error retrieving FMP global news: {e}"


def get_insider_transactions(ticker: str):
    try:
        rows = _rows(
            _call(
                "insiderTrades",
                {
                    "endpoint": "search-insider-trades",
                    "symbol": ticker.upper(),
                    "limit": 20,
                },
            )
        )
        filtered = [row for row in rows if str(row.get("symbol") or "").upper() == ticker.upper()]
        if not filtered:
            filtered = rows
        if not filtered:
            return f"No insider transaction data found for symbol '{ticker}'"
        normalized = [
            {
                "filingDate": row.get("filingDate"),
                "transactionDate": row.get("transactionDate"),
                "reportingName": row.get("reportingName"),
                "typeOfOwner": row.get("typeOfOwner"),
                "transactionType": row.get("transactionType"),
                "acquisitionOrDisposition": row.get("acquisitionOrDisposition"),
                "securityName": row.get("securityName"),
                "securitiesTransacted": row.get("securitiesTransacted"),
                "securitiesOwned": row.get("securitiesOwned"),
                "price": row.get("price"),
                "url": row.get("url"),
            }
            for row in _sorted_rows(filtered, key="filingDate", reverse=True)
        ]
        return _header(f"Insider Transactions for {ticker.upper()}") + _to_csv_block(normalized)
    except Exception as e:
        return f"Error retrieving FMP insider transactions for {ticker}: {e}"


def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int):
    normalized_indicator = indicator.strip().lower()
    if normalized_indicator not in _INDICATOR_MAP:
        raise ValueError(
            f"Indicator {indicator} is not supported by FMP. Please choose from: {list(_INDICATOR_MAP.keys())}"
        )

    endpoint, period_length, result_key = _INDICATOR_MAP[normalized_indicator]
    start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)
    try:
        rows = _rows(
            _call(
                "technicalIndicators",
                {
                    "endpoint": endpoint,
                    "symbol": symbol.upper(),
                    "periodLength": period_length,
                    "timeframe": "1day",
                    "from_date": start_dt.strftime("%Y-%m-%d"),
                    "to_date": curr_date,
                },
            )
        )
        rows = _sorted_rows(rows, reverse=True)
        if not rows:
            return f"No {indicator} data found for symbol '{symbol}'"
        values = []
        for row in rows:
            date_value = str(row.get("date") or "").split(" ")[0]
            values.append(f"{date_value}: {row.get(result_key, 'N/A')}")
        return (
            f"## {normalized_indicator} values from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
            + "\n".join(values)
            + f"\n\nComputed by FMP MCP using endpoint '{endpoint}' with periodLength={period_length}."
        )
    except Exception as e:
        return f"Error retrieving FMP indicator {indicator} for {symbol}: {e}"
