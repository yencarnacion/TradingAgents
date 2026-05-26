from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import get_config
from .mcp_support import MCPToolError, call_tool, market_data_mcp_url

_MASSIVE_BASE_URL = os.getenv("MASSIVE_BASE_URL", "https://api.massive.com")
_ET = ZoneInfo("America/New_York")
_SESSION_WINDOWS = {
    "premarket": ((4, 0), (9, 29)),
    "regular": ((9, 30), (16, 0)),
    "postmarket": ((16, 1), (19, 59)),
    "afterhours": ((16, 1), (19, 59)),
}
_DEFAULT_SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU")
_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
    ),
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes. "
        "Tips: Confirm with other indicators in low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. "
        "Usage: Use crossovers with the MACD line to trigger trades. "
        "Tips: Should be part of a broader strategy to avoid false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. "
        "Usage: Visualize momentum strength and spot divergence early. "
        "Tips: Can be volatile; complement with additional filters in fast-moving markets."
    ),
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
        "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
    ),
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
        "Usage: Acts as a dynamic benchmark for price movement. "
        "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
        "Usage: Signals potential overbought conditions and breakout zones. "
        "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
        "Usage: Indicates potential oversold conditions. "
        "Tips: Use additional analysis to avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. "
        "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
        "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
    ),
    "vwma": (
        "VWMA: A moving average weighted by volume. "
        "Usage: Confirm trends by integrating price action with volume data. "
        "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
    ),
    "mfi": (
        "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
        "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
        "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
    ),
}


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


def _snapshot_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = payload.get("results") or []
    if isinstance(results, dict):
        if "ticker" in results:
            return [results]
        tickers = results.get("tickers")
        if isinstance(tickers, list):
            return [row for row in tickers if isinstance(row, dict)]
    if isinstance(results, list):
        return [row for row in results if isinstance(row, dict)]
    return []


def _timestamp_ms_to_et(timestamp_ms: Any) -> Optional[datetime]:
    if timestamp_ms in (None, ""):
        return None
    return datetime.fromtimestamp(int(float(timestamp_ms)) / 1000, tz=ZoneInfo("UTC")).astimezone(_ET)


def _session_label(session: str) -> str:
    normalized = session.strip().lower()
    return "postmarket" if normalized == "afterhours" else normalized


def _time_in_window(ts: datetime, start: tuple[int, int], end: tuple[int, int]) -> bool:
    current = (ts.hour, ts.minute)
    return start <= current <= end


def _session_window_text(session: str) -> str:
    start, end = _SESSION_WINDOWS[_session_label(session)]
    return f"{start[0]:02d}:{start[1]:02d}-{end[0]:02d}:{end[1]:02d}"


def _candidate_trade_dates(trade_date: str, max_lookback_days: int = 7) -> List[str]:
    requested = datetime.strptime(trade_date, "%Y-%m-%d")
    return [
        (requested - timedelta(days=offset)).strftime("%Y-%m-%d")
        for offset in range(max_lookback_days + 1)
    ]


def _find_latest_available_rows(
    requested_trade_date: str,
    fetch_rows: Callable[[str], List[Dict[str, Any]]],
    max_lookback_days: int = 7,
) -> tuple[str, List[Dict[str, Any]]]:
    for candidate_trade_date in _candidate_trade_dates(requested_trade_date, max_lookback_days=max_lookback_days):
        rows = fetch_rows(candidate_trade_date)
        if rows:
            return candidate_trade_date, rows
    return requested_trade_date, []


def _effective_trade_date_note(requested_trade_date: str, effective_trade_date: str) -> str:
    if effective_trade_date == requested_trade_date:
        return f"# Requested trade date: {requested_trade_date}\n\n"
    return (
        f"# Requested trade date: {requested_trade_date}\n"
        f"# Using latest completed session: {effective_trade_date}\n\n"
    )


def _normalize_aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for row in rows:
        timestamp_ms = row.get("t")
        timestamp_et = _timestamp_ms_to_et(timestamp_ms)
        normalized.append(
            {
                "timestamp_et": timestamp_et.strftime("%Y-%m-%d %H:%M:%S ET") if timestamp_et else None,
                "open": row.get("o"),
                "high": row.get("h"),
                "low": row.get("l"),
                "close": row.get("c"),
                "volume": row.get("v"),
                "vwap": row.get("vw"),
                "transactions": row.get("n"),
                "timestamp_ms": int(float(timestamp_ms)) if timestamp_ms not in (None, "") else None,
            }
        )
    return normalized


def _ticker_snapshot_rows(symbols: List[str]) -> List[Dict[str, Any]]:
    if len(symbols) == 1:
        payload = _request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbols[0]}")
    else:
        payload = _request(
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            params={"tickers": ",".join(symbols)},
        )
    return _snapshot_rows(payload)


def _flatten_snapshot_row(row: Dict[str, Any]) -> Dict[str, Any]:
    session = row.get("session") or {}
    last_trade = row.get("lastTrade") or row.get("last_trade") or {}
    prev_day = row.get("prevDay") or row.get("prev_day") or {}
    return {
        "ticker": row.get("ticker"),
        "session_open": session.get("open"),
        "session_high": session.get("high"),
        "session_low": session.get("low"),
        "session_close": session.get("close"),
        "session_change": session.get("change"),
        "session_change_percent": session.get("change_percent"),
        "session_volume": session.get("volume"),
        "last_trade_price": last_trade.get("p") or last_trade.get("price"),
        "last_trade_size": last_trade.get("s") or last_trade.get("size"),
        "last_trade_timestamp": last_trade.get("t") or last_trade.get("timestamp"),
        "prev_close": prev_day.get("close"),
        "prev_volume": prev_day.get("volume"),
    }


def _spot_price_from_snapshot(symbol: str) -> Optional[float]:
    rows = _ticker_snapshot_rows([symbol.upper()])
    if not rows:
        return None
    row = rows[0]
    session = row.get("session") or {}
    last_trade = row.get("lastTrade") or row.get("last_trade") or {}
    for value in (session.get("close"), last_trade.get("p"), last_trade.get("price"), session.get("open")):
        if value not in (None, ""):
            return float(value)
    return None


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


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _indicator_frame(symbol: str, curr_date: str, look_back_days: int) -> pd.DataFrame:
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    warmup_days = max(260, look_back_days + 220)
    start_date = (end_dt - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    payload = _request(
        f"/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start_date}/{curr_date}",
        params={"adjusted": "true", "sort": "asc", "limit": 5000},
    )
    rows = _rows(payload)
    if not rows:
        raise RuntimeError(f"No Massive price history returned for {symbol} between {start_date} and {curr_date}")

    normalized = []
    for row in rows:
        if row.get("t") not in (None, ""):
            trade_date = datetime.fromtimestamp(int(float(row["t"])) / 1000, tz=ZoneInfo("UTC")).astimezone(_ET).date().isoformat()
        else:
            trade_date = str(row.get("date") or "").split(" ")[0]
        normalized.append(
            {
                "date": trade_date,
                "open": row.get("o", row.get("open")),
                "high": row.get("h", row.get("high")),
                "low": row.get("l", row.get("low")),
                "close": row.get("c", row.get("close")),
                "volume": row.get("v", row.get("volume")),
                "vwap": row.get("vw", row.get("vwap")),
            }
        )

    frame = pd.DataFrame(normalized)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "vwap"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["date", "open", "high", "low", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    if frame.empty:
        raise RuntimeError(f"Massive price history for {symbol} was empty after normalization")

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"].fillna(0)
    typical_price = (high + low + close) / 3

    frame["close_50_sma"] = close.rolling(window=50, min_periods=50).mean()
    frame["close_200_sma"] = close.rolling(window=200, min_periods=200).mean()
    frame["close_10_ema"] = close.ewm(span=10, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema12 - ema26
    frame["macds"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macdh"] = frame["macd"] - frame["macds"]
    frame["rsi"] = _compute_rsi(close, 14)
    frame["boll"] = close.rolling(window=20, min_periods=20).mean()
    boll_std = close.rolling(window=20, min_periods=20).std()
    frame["boll_ub"] = frame["boll"] + (boll_std * 2)
    frame["boll_lb"] = frame["boll"] - (boll_std * 2)
    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    frame["atr"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    volume_sum = volume.rolling(window=20, min_periods=20).sum()
    frame["vwma"] = (close * volume).rolling(window=20, min_periods=20).sum() / volume_sum.where(volume_sum != 0)
    raw_money_flow = typical_price * volume
    tp_delta = typical_price.diff()
    positive_flow = raw_money_flow.where(tp_delta > 0, 0.0)
    negative_flow = raw_money_flow.where(tp_delta < 0, 0.0).abs()
    positive_sum = positive_flow.rolling(window=14, min_periods=14).sum()
    negative_sum = negative_flow.rolling(window=14, min_periods=14).sum()
    money_ratio = positive_sum / negative_sum.replace(0, pd.NA)
    frame["mfi"] = 100 - (100 / (1 + money_ratio))
    frame.loc[(negative_sum == 0) & (positive_sum > 0), "mfi"] = 100.0
    frame["date_str"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


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


def get_indicators(symbol: str, indicator: str, curr_date: str, look_back_days: int) -> str:
    if indicator not in _INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(_INDICATOR_DESCRIPTIONS.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - timedelta(days=look_back_days)

    try:
        frame = _indicator_frame(symbol, curr_date, look_back_days)
        value_map = {}
        for _, row in frame.iterrows():
            value = row.get(indicator)
            value_map[row["date_str"]] = "N/A" if pd.isna(value) else str(round(float(value), 6))

        lines = []
        current_dt = curr_date_dt
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            lines.append(f"{date_str}: {value_map.get(date_str, 'N/A: Not a trading day (weekend or holiday)')}")
            current_dt -= timedelta(days=1)

        return (
            f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
            + "\n".join(lines)
            + "\n\n"
            + _INDICATOR_DESCRIPTIONS[indicator]
        )
    except Exception as e:
        return f"Error retrieving Massive indicator data for {symbol}: {e}"


def get_intraday_bars(symbol: str, trade_date: str, multiplier: int = 5, timespan: str = "minute", limit: int = 5000):
    try:
        def fetch_rows(candidate_trade_date: str) -> List[Dict[str, Any]]:
            payload = _request(
                f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{candidate_trade_date}/{candidate_trade_date}",
                params={"adjusted": "true", "sort": "asc", "limit": limit},
            )
            return _rows(payload)

        effective_trade_date, rows = _find_latest_available_rows(trade_date, fetch_rows)
        if not rows:
            return f"No intraday {timespan} bars found for symbol '{symbol}' on {trade_date}"
        normalized = _normalize_aggregate_rows(rows)
        return (
            _header(f"Intraday {multiplier}-{timespan} bars for {symbol.upper()} on {effective_trade_date}")
            + _effective_trade_date_note(trade_date, effective_trade_date)
            + _to_csv_block(normalized)
        )
    except Exception as e:
        return f"Error retrieving Massive intraday bars for {symbol}: {e}"


def get_session_bars(symbol: str, trade_date: str, session: str = "premarket", multiplier: int = 5, timespan: str = "minute", limit: int = 5000):
    try:
        normalized_session = _session_label(session)
        if normalized_session not in _SESSION_WINDOWS:
            raise ValueError(f"Unsupported session '{session}'. Choose from: {', '.join(sorted(_SESSION_WINDOWS))}")
        start, end = _SESSION_WINDOWS[normalized_session]

        def fetch_rows(candidate_trade_date: str) -> List[Dict[str, Any]]:
            payload = _request(
                f"/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{candidate_trade_date}/{candidate_trade_date}",
                params={"adjusted": "true", "sort": "asc", "limit": limit},
            )
            filtered_rows = []
            for row in _rows(payload):
                timestamp_et = _timestamp_ms_to_et(row.get("t"))
                if timestamp_et is None or timestamp_et.strftime("%Y-%m-%d") != candidate_trade_date:
                    continue
                if _time_in_window(timestamp_et, start, end):
                    filtered_rows.append(row)
            return filtered_rows

        effective_trade_date, filtered_rows = _find_latest_available_rows(trade_date, fetch_rows)
        if not filtered_rows:
            return f"No {normalized_session} {timespan} bars found for symbol '{symbol}' on {trade_date}"
        normalized = _normalize_aggregate_rows(filtered_rows)
        return (
            _header(f"{normalized_session.title()} {multiplier}-{timespan} bars for {symbol.upper()} on {effective_trade_date}")
            + _effective_trade_date_note(trade_date, effective_trade_date)
            + f"# Session window (America/New_York): {_session_window_text(normalized_session)}\n\n"
            + _to_csv_block(normalized)
        )
    except Exception as e:
        return f"Error retrieving Massive session bars for {symbol}: {e}"


def get_ticker_snapshot(symbol: str):
    try:
        rows = _ticker_snapshot_rows([symbol.upper()])
        if not rows:
            return f"No snapshot found for symbol '{symbol}'"
        return _header(f"Ticker snapshot for {symbol.upper()}") + _to_csv_block([_flatten_snapshot_row(rows[0])])
    except Exception as e:
        return f"Error retrieving Massive ticker snapshot for {symbol}: {e}"


def get_last_trade(symbol: str):
    try:
        payload = _request(f"/v2/last/trade/{symbol.upper()}")
        rows = _rows(payload)
        if not rows:
            return f"No last-trade data found for symbol '{symbol}'"
        row = rows[0]
        normalized = {
            "ticker": symbol.upper(),
            "price": row.get("p") or row.get("price"),
            "size": row.get("s") or row.get("size"),
            "exchange": row.get("x") or row.get("exchange"),
            "timestamp_ms": row.get("t") or row.get("timestamp"),
        }
        return _header(f"Last trade for {symbol.upper()}") + _to_csv_block([normalized])
    except Exception as e:
        return f"Error retrieving Massive last trade for {symbol}: {e}"


def get_nbbo_quotes(symbol: str, limit: int = 20):
    try:
        payload = _request(
            f"/v3/quotes/{symbol.upper()}",
            params={"limit": limit, "sort": "timestamp", "order": "desc"},
        )
        rows = []
        for row in _rows(payload)[:limit]:
            rows.append(
                {
                    "ticker": row.get("ticker") or symbol.upper(),
                    "bid_price": row.get("bp"),
                    "bid_size": row.get("bs"),
                    "ask_price": row.get("ap"),
                    "ask_size": row.get("as"),
                    "spread": (
                        round(float(row.get("ap")) - float(row.get("bp")), 6)
                        if row.get("ap") not in (None, "") and row.get("bp") not in (None, "")
                        else None
                    ),
                    "sip_timestamp": row.get("sip_timestamp") or row.get("t"),
                }
            )
        if not rows:
            return f"No NBBO quote data found for symbol '{symbol}'"
        return _header(f"NBBO quotes for {symbol.upper()}") + _to_csv_block(rows)
    except Exception as e:
        return f"Error retrieving Massive NBBO quotes for {symbol}: {e}"


def get_market_regime(curr_date: str, benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM")):
    try:
        effective_trade_date, grouped = _find_latest_available_rows(
            curr_date,
            lambda candidate_trade_date: _rows(_request(f"/v2/aggs/grouped/locale/us/market/stocks/{candidate_trade_date}")),
        )
        advancers = decliners = unchanged = up_volume = down_volume = 0
        for row in grouped:
            open_price = row.get("o")
            close_price = row.get("c")
            volume = row.get("v") or 0
            if open_price in (None, "") or close_price in (None, ""):
                continue
            open_price = float(open_price)
            close_price = float(close_price)
            volume = int(float(volume)) if volume not in (None, "") else 0
            if close_price > open_price:
                advancers += 1
                up_volume += volume
            elif close_price < open_price:
                decliners += 1
                down_volume += volume
            else:
                unchanged += 1
        breadth_rows = [{
            "advancers": advancers,
            "decliners": decliners,
            "unchanged": unchanged,
            "advance_decline_ratio": round(advancers / decliners, 4) if decliners else None,
            "up_volume": up_volume,
            "down_volume": down_volume,
        }]
        regime_symbols = [symbol.upper() for symbol in benchmark_symbols] + [symbol for symbol in _DEFAULT_SECTOR_ETFS if symbol not in benchmark_symbols]
        snapshots = [_flatten_snapshot_row(row) for row in _ticker_snapshot_rows(regime_symbols)]
        sections = [
            _header(f"Market regime for {effective_trade_date}").rstrip(),
            "",
            _effective_trade_date_note(curr_date, effective_trade_date).strip(),
            "## Market breadth",
            _to_csv_block(breadth_rows).strip(),
            "",
            "## Benchmark and sector snapshots",
            _to_csv_block(snapshots).strip(),
        ]
        try:
            from . import fmp
            vix_rows = fmp._rows(fmp._call("indexes", {"endpoint": "index-quote-short", "symbol": "^VIX"}))
            if vix_rows:
                sections.extend(["", "## VIX fallback", fmp._to_csv_block(vix_rows).strip()])
        except Exception:
            pass
        return "\n".join(sections).strip() + "\n"
    except Exception as e:
        return f"Error retrieving Massive market regime for {curr_date}: {e}"


def get_options_chain(symbol: str, trade_date: str, expiration_date: Optional[str] = None, contract_type: Optional[str] = None, strike_window: int = 5, limit: int = 25):
    try:
        spot_price = _spot_price_from_snapshot(symbol)

        def fetch_contract_rows(candidate_trade_date: str) -> List[Dict[str, Any]]:
            params: Dict[str, Any] = {
                "underlying_ticker": symbol.upper(),
                "limit": max(limit * 3, 25),
                "sort": "expiration_date",
                "order": "asc",
                "as_of": candidate_trade_date,
            }
            if expiration_date:
                params["expiration_date"] = expiration_date
            if contract_type:
                params["contract_type"] = contract_type.lower()
            return _rows(_request("/v3/reference/options/contracts", params=params))

        effective_trade_date, contracts = _find_latest_available_rows(trade_date, fetch_contract_rows)
        if not contracts:
            return f"No options contracts found for symbol '{symbol}' on {trade_date}"
        if spot_price is not None:
            within_window = [
                row for row in contracts
                if row.get("strike_price") not in (None, "") and abs(float(row.get("strike_price")) - spot_price) <= strike_window
            ]
            ranked = sorted(within_window or contracts, key=lambda row: (
                abs(float(row.get("strike_price") or 0) - spot_price),
                str(row.get("expiration_date") or ""),
                float(row.get("strike_price") or 0),
            ))
        else:
            ranked = contracts
        selected = ranked[:limit]
        normalized_rows = []
        for row in selected:
            contract_symbol = row.get("ticker")
            prev_payload = _rows(_request(f"/v2/aggs/ticker/{contract_symbol}/prev"))
            prev_row = prev_payload[0] if prev_payload else {}
            day_effective_trade_date, day_payload = _find_latest_available_rows(
                trade_date,
                lambda candidate_trade_date: _rows(
                    _request(f"/v2/aggs/ticker/{contract_symbol}/range/1/day/{candidate_trade_date}/{candidate_trade_date}")
                ),
            )
            day_row = day_payload[0] if day_payload else {}
            normalized_rows.append(
                {
                    "ticker": contract_symbol,
                    "underlying_ticker": symbol.upper(),
                    "spot_price": _clean_value(spot_price),
                    "expiration_date": row.get("expiration_date"),
                    "contract_type": row.get("contract_type"),
                    "strike_price": row.get("strike_price"),
                    "shares_per_contract": row.get("shares_per_contract"),
                    "prev_close": prev_row.get("c"),
                    "prev_volume": prev_row.get("v"),
                    "trade_date_effective": day_effective_trade_date,
                    "trade_date_open": day_row.get("o"),
                    "trade_date_high": day_row.get("h"),
                    "trade_date_low": day_row.get("l"),
                    "trade_date_close": day_row.get("c"),
                    "trade_date_volume": day_row.get("v"),
                }
            )
        return (
            _header(f"Options chain for {symbol.upper()} on {effective_trade_date}")
            + _effective_trade_date_note(trade_date, effective_trade_date)
            + _to_csv_block(normalized_rows)
        )
    except Exception as e:
        return f"Error retrieving Massive options chain for {symbol}: {e}"


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
