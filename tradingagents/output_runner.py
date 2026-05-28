from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.fmp import get_recent_earnings_anchor_data
from tradingagents.dataflows.interface import get_vendor, route_to_vendor
from tradingagents.dataflows.utils import safe_ticker_component
from zoneinfo import ZoneInfo


DEFAULT_TICKER = "SPY"
DEFAULT_PUBLIC_HOST = os.getenv("TICKER_AGENTS_PUBLIC_HOST")
DEFAULT_PORT = int(os.getenv("TICKER_AGENTS_OUTPUT_PORT", "8765"))
NEW_YORK_TZ = ZoneInfo("America/New_York")
FINAL_BEGIN = "=== FINAL_DECISION_MARKDOWN_BEGIN ==="
FINAL_END = "=== FINAL_DECISION_MARKDOWN_END ==="
STATE_BEGIN = "=== FINAL_STATE_REPORTS_JSON_BEGIN ==="
STATE_END = "=== FINAL_STATE_REPORTS_JSON_END ==="
STACK_EXAMPLES = {
    "grounded": "run_grounded_stack.py",
    "fmp": "run_fmp_mcp_stack.py",
}
TECHNICAL_INDICATORS: tuple[tuple[str, str], ...] = (
    ("close_10_ema", "10 EMA"),
    ("close_50_sma", "50 SMA"),
    ("close_200_sma", "200 SMA"),
    ("macd", "MACD"),
    ("macds", "MACD Signal"),
    ("macdh", "MACD Histogram"),
    ("rsi", "RSI"),
    ("boll", "Bollinger Middle"),
    ("boll_ub", "Bollinger Upper Band"),
    ("boll_lb", "Bollinger Lower Band"),
    ("atr", "ATR"),
    ("vwma", "VWMA"),
)
TECHNICAL_CHART_LOOKBACK_DAYS = 420


@dataclass
class RunPaths:
    run_dir: Path
    console_txt: Path
    metadata_json: Path
    live_html: Path
    index_html: Path
    final_md: Path
    final_html: Path


@dataclass(frozen=True)
class ReportArtifactSpec:
    slug: str
    title: str
    category: str
    extractor: Callable[[dict[str, Any]], Optional[str]]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "run"


def resolve_run_request(
    ticker: Optional[str], analysis_date: Optional[str], *, today: Optional[date] = None
) -> tuple[str, str]:
    today = today or date.today()
    resolved_ticker = (ticker or DEFAULT_TICKER).strip().upper()
    resolved_date = (analysis_date or today.isoformat()).strip()
    datetime.strptime(resolved_date, "%Y-%m-%d")
    return resolved_ticker, resolved_date


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_run_slug(ticker: str, analysis_date: str, started_at: datetime) -> str:
    host = slugify(socket.gethostname())
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    return slugify(f"{ticker}-{analysis_date}-{stamp}-{host}-pid{os.getpid()}")


def build_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        run_dir=run_dir,
        console_txt=run_dir / "console.txt",
        metadata_json=run_dir / "metadata.json",
        live_html=run_dir / "live.html",
        index_html=run_dir / "index.html",
        final_md=run_dir / "final.md",
        final_html=run_dir / "final.html",
    )


def build_app_command(repo_root: Path, ticker: str, analysis_date: str, stack: str) -> list[str]:
    example_name = STACK_EXAMPLES[stack]
    return [
        str(repo_root / ".venv" / "bin" / "python"),
        str(repo_root / "examples" / example_name),
        ticker,
        analysis_date,
    ]


def extract_final_decision(log_text: str) -> Optional[str]:
    start = log_text.rfind(FINAL_BEGIN)
    end = log_text.rfind(FINAL_END)
    if start == -1 or end == -1 or end <= start:
        return None
    body = log_text[start + len(FINAL_BEGIN):end].strip()
    return body or None


def extract_state_reports(log_text: str) -> Optional[dict[str, Any]]:
    start = log_text.rfind(STATE_BEGIN)
    end = log_text.rfind(STATE_END)
    if start == -1 or end == -1 or end <= start:
        return None
    body = log_text[start + len(STATE_BEGIN):end].strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_nested_text(state: dict[str, Any], *keys: str) -> Optional[str]:
    current: Any = state
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if not isinstance(current, str):
        return None
    text = current.strip()
    return text or None


REPORT_ARTIFACT_SPECS: tuple[ReportArtifactSpec, ...] = (
    ReportArtifactSpec("market-report", "Market Report", "analyst", lambda state: _extract_nested_text(state, "market_report")),
    ReportArtifactSpec("sentiment-report", "Sentiment Report", "analyst", lambda state: _extract_nested_text(state, "sentiment_report")),
    ReportArtifactSpec("news-report", "News Report", "analyst", lambda state: _extract_nested_text(state, "news_report")),
    ReportArtifactSpec("fundamentals-report", "Fundamentals Report", "analyst", lambda state: _extract_nested_text(state, "fundamentals_report")),
    ReportArtifactSpec("bullish-report", "Bullish Report", "debate", lambda state: _extract_nested_text(state, "investment_debate_state", "bull_history")),
    ReportArtifactSpec("bearish-report", "Bearish Report", "debate", lambda state: _extract_nested_text(state, "investment_debate_state", "bear_history")),
    ReportArtifactSpec("research-manager-report", "Research Manager Report", "debate", lambda state: _extract_nested_text(state, "investment_debate_state", "judge_decision")),
    ReportArtifactSpec("investment-plan", "Investment Plan", "decision", lambda state: _extract_nested_text(state, "investment_plan")),
    ReportArtifactSpec("trader-report", "Trader Report", "decision", lambda state: _extract_nested_text(state, "trader_investment_decision")),
    ReportArtifactSpec("aggressive-risk-report", "Aggressive Risk Report", "risk", lambda state: _extract_nested_text(state, "risk_debate_state", "aggressive_history")),
    ReportArtifactSpec("conservative-risk-report", "Conservative Risk Report", "risk", lambda state: _extract_nested_text(state, "risk_debate_state", "conservative_history")),
    ReportArtifactSpec("neutral-risk-report", "Neutral Risk Report", "risk", lambda state: _extract_nested_text(state, "risk_debate_state", "neutral_history")),
    ReportArtifactSpec("portfolio-manager-report", "Portfolio Manager Report", "risk", lambda state: _extract_nested_text(state, "risk_debate_state", "judge_decision")),
)

REPORT_CATEGORY_TITLES: dict[str, str] = {
    "technical": "Technical reports",
    "analyst": "Analyst reports",
    "debate": "Debate reports",
    "decision": "Decision reports",
    "risk": "Risk reports",
}

REPORT_CATEGORY_ORDER: tuple[str, ...] = tuple(REPORT_CATEGORY_TITLES)


def state_log_path(ticker: str, analysis_date: str) -> Path:
    return (
        Path.home()
        / ".tradingagents"
        / "logs"
        / ticker
        / "TradingAgentsStrategy_logs"
        / f"full_states_log_{analysis_date}.json"
    )


def state_log_path_from_config(ticker: str, analysis_date: str, config: dict[str, Any]) -> Path:
    results_dir = Path(config.get("results_dir") or Path.home() / ".tradingagents" / "logs")
    return (
        results_dir
        / safe_ticker_component(ticker)
        / "TradingAgentsStrategy_logs"
        / f"full_states_log_{analysis_date}.json"
    )


def load_state_log(path: Path, *, min_modified_at: Optional[float] = None) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    if min_modified_at is not None:
        try:
            if path.stat().st_mtime < min_modified_at:
                return None
        except OSError:
            return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def append_artifact_warning(metadata: dict[str, Any], message: str) -> None:
    existing = metadata.get("artifact_warning")
    if existing and message in existing.split("; "):
        return
    metadata["artifact_warning"] = f"{existing}; {message}" if existing else message


def format_timestamp_for_display(raw: Optional[str]) -> str:
    if not raw:
        return "—"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return raw
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed.astimezone(NEW_YORK_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def _join_public_url(base_url: Optional[str], filename: str) -> Optional[str]:
    if not base_url:
        return filename
    return f"{base_url.rstrip('/')}/{filename}"


def detect_public_host(override: Optional[str] = None) -> str:
    """Best-effort host for URLs printed to the terminal.

    HTML and metadata use relative links so they follow whichever host the
    browser opened. This is only for the initial convenience URL printed by
    the runner.
    """
    if override:
        return override
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
            if host and not host.startswith("127."):
                return host
    except OSError:
        pass
    try:
        host = socket.gethostbyname(socket.gethostname())
        if host and not host.startswith("127."):
            return host
    except OSError:
        pass
    return "localhost"


def public_run_url(port: int, slug: str, filename: str, host: Optional[str] = None) -> str:
    return f"http://{detect_public_host(host)}:{port}/{slug}/{filename}"


def summarize_report_artifacts(artifacts: list[dict[str, str]]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for category in REPORT_CATEGORY_ORDER:
        members = [artifact for artifact in artifacts if artifact.get("category") == category]
        if not members:
            continue
        groups.append(
            {
                "key": category,
                "title": REPORT_CATEGORY_TITLES.get(category, category.title()),
                "count": len(members),
                "artifacts": members,
            }
        )
    return {
        "count": len(artifacts),
        "groups": groups,
    }


def _build_stack_config(stack: str) -> dict[str, Any]:
    if stack == "fmp":
        from examples.run_fmp_mcp_stack import build_config

        return build_config()
    if stack == "grounded":
        from examples.run_grounded_stack import build_config

        return build_config()
    raise ValueError(f"Unknown stack: {stack}")


def _technical_artifact_payload(
    *,
    slug: str,
    title: str,
    md_path: Path,
    html_path: Path,
    public_base_url: Optional[str],
) -> dict[str, str]:
    return {
        "slug": slug,
        "title": title,
        "category": "technical",
        "markdown_path": md_path.name,
        "html_path": html_path.name,
        "markdown_url": _join_public_url(public_base_url, md_path.name) or "",
        "html_url": _join_public_url(public_base_url, html_path.name) or "",
    }


def _parse_csv_records(tool_output: str) -> list[dict[str, str]]:
    csv_lines = [line for line in tool_output.splitlines() if line.strip() and not line.startswith("#")]
    if not csv_lines:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(csv_lines))))


def _coerce_numeric_series(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _series_points(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        return []
    subset = frame[["time", column]].dropna()
    return [
        {"time": row["time"], "value": round(float(row[column]), 6)}
        for _, row in subset.iterrows()
    ]


def _histogram_points(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if column not in frame.columns:
        return []
    subset = frame[["time", column]].dropna()
    return [
        {
            "time": row["time"],
            "value": round(float(row[column]), 6),
            "color": "#22c55e" if float(row[column]) >= 0 else "#ef4444",
        }
        for _, row in subset.iterrows()
    ]


def load_recent_earnings_anchor(ticker: str, analysis_date: str, stack: str) -> dict[str, Any]:
    config = _build_stack_config(stack)
    set_config(config)
    try:
        anchor = get_recent_earnings_anchor_data(ticker, analysis_date)
    except Exception:
        return {}
    return anchor or {}


def _compute_avwap_from_anchor(frame: pd.DataFrame, anchor_date: str) -> pd.Series:
    if not anchor_date or frame.empty:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    if "date" not in frame.columns or "volume" not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")

    anchor_mask = frame["date"] >= anchor_date
    price_basis = frame.get("vwap")
    if price_basis is None:
        price_basis = (frame["high"] + frame["low"] + frame["close"]) / 3
    weighted = price_basis * frame["volume"]
    anchor_weighted = weighted.where(anchor_mask)
    anchor_volume = frame["volume"].where(anchor_mask)
    cumulative_volume = anchor_volume.cumsum()
    avwap = anchor_weighted.cumsum() / cumulative_volume.replace(0, pd.NA)
    avwap = avwap.where(anchor_mask)
    return avwap


def load_massive_chart_data(
    ticker: str,
    analysis_date: str,
    *,
    stack: str,
    look_back_days: int = TECHNICAL_CHART_LOOKBACK_DAYS,
) -> dict[str, Any]:
    config = _build_stack_config(stack)
    config.setdefault("data_vendors", {})["core_stock_apis"] = "massive"
    config.setdefault("tool_vendors", {})["get_stock_data"] = "massive"
    set_config(config)

    end_dt = datetime.strptime(analysis_date, "%Y-%m-%d")
    start_date = (end_dt - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    stock_csv = route_to_vendor("get_stock_data", ticker, start_date, analysis_date)
    rows = _parse_csv_records(stock_csv)
    if not rows:
        raise ValueError(f"No Massive OHLCV data returned for {ticker} between {start_date} and {analysis_date}")

    frame = pd.DataFrame(rows)
    required_columns = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Massive OHLCV data missing required columns: {', '.join(missing)}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = _coerce_numeric_series(frame, ["open", "high", "low", "close", "volume", "vwap"])
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Massive OHLCV data for {ticker} was empty after normalization")
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")

    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"].fillna(0)

    frame["ema10"] = close.ewm(span=10, adjust=False).mean()
    frame["sma50"] = close.rolling(window=50, min_periods=50).mean()
    frame["sma200"] = close.rolling(window=200, min_periods=200).mean()
    frame["boll_mid"] = close.rolling(window=20, min_periods=20).mean()
    boll_std = close.rolling(window=20, min_periods=20).std()
    frame["boll_upper"] = frame["boll_mid"] + (boll_std * 2)
    frame["boll_lower"] = frame["boll_mid"] - (boll_std * 2)
    volume_sum = volume.rolling(window=20, min_periods=20).sum()
    frame["vwma20"] = (close * volume).rolling(window=20, min_periods=20).sum() / volume_sum.where(volume_sum != 0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]
    frame["rsi14"] = _compute_rsi(close, 14)

    earnings_anchor = load_recent_earnings_anchor(ticker, analysis_date, stack)
    frame["avwap_from_earnings"] = _compute_avwap_from_anchor(frame, str(earnings_anchor.get("anchor_date") or ""))

    prev_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    frame["atr14"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    frame["time"] = frame["date"]

    candles = [
        {
            "time": row["time"],
            "open": round(float(row["open"]), 6),
            "high": round(float(row["high"]), 6),
            "low": round(float(row["low"]), 6),
            "close": round(float(row["close"]), 6),
        }
        for _, row in frame.iterrows()
    ]
    latest = frame.iloc[-1]
    indicator_snapshot = {
        "close": round(float(latest["close"]), 6),
        "ema10": round(float(latest["ema10"]), 6) if pd.notna(latest["ema10"]) else None,
        "sma50": round(float(latest["sma50"]), 6) if pd.notna(latest["sma50"]) else None,
        "sma200": round(float(latest["sma200"]), 6) if pd.notna(latest["sma200"]) else None,
        "boll_upper": round(float(latest["boll_upper"]), 6) if pd.notna(latest["boll_upper"]) else None,
        "boll_mid": round(float(latest["boll_mid"]), 6) if pd.notna(latest["boll_mid"]) else None,
        "boll_lower": round(float(latest["boll_lower"]), 6) if pd.notna(latest["boll_lower"]) else None,
        "vwma20": round(float(latest["vwma20"]), 6) if pd.notna(latest["vwma20"]) else None,
        "avwap_from_earnings": round(float(latest["avwap_from_earnings"]), 6) if pd.notna(latest["avwap_from_earnings"]) else None,
        "rsi14": round(float(latest["rsi14"]), 6) if pd.notna(latest["rsi14"]) else None,
        "macd": round(float(latest["macd"]), 6) if pd.notna(latest["macd"]) else None,
        "macd_signal": round(float(latest["macd_signal"]), 6) if pd.notna(latest["macd_signal"]) else None,
        "macd_hist": round(float(latest["macd_hist"]), 6) if pd.notna(latest["macd_hist"]) else None,
        "atr14": round(float(latest["atr14"]), 6) if pd.notna(latest["atr14"]) else None,
    }
    return {
        "source_vendor": "massive",
        "start_date": start_date,
        "end_date": analysis_date,
        "earnings_anchor": earnings_anchor,
        "candles": candles,
        "ema10": _series_points(frame, "ema10"),
        "sma50": _series_points(frame, "sma50"),
        "sma200": _series_points(frame, "sma200"),
        "boll_upper": _series_points(frame, "boll_upper"),
        "boll_mid": _series_points(frame, "boll_mid"),
        "boll_lower": _series_points(frame, "boll_lower"),
        "vwma20": _series_points(frame, "vwma20"),
        "avwap_from_earnings": _series_points(frame, "avwap_from_earnings"),
        "rsi14": _series_points(frame, "rsi14"),
        "macd": _series_points(frame, "macd"),
        "macd_signal": _series_points(frame, "macd_signal"),
        "macd_hist": _histogram_points(frame, "macd_hist"),
        "indicator_snapshot": indicator_snapshot,
    }


def render_technical_chart_html(title: str, ticker: str, analysis_date: str, chart_data: dict[str, Any]) -> str:
    escaped_title = html.escape(title)
    escaped_ticker = html.escape(ticker)
    escaped_date = html.escape(analysis_date)
    payload = json.dumps(chart_data, separators=(",", ":")).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; background:#0b1020; color:#e5e7eb; font-family:Inter,system-ui,sans-serif; }
    main { max-width: 1280px; margin: 0 auto; padding: 24px; }
    a { color:#93c5fd; }
    .hero, .panel { background:#111827; border:1px solid #1f2937; border-radius:16px; padding:20px; margin-bottom:18px; }
    .meta { color:#94a3b8; display:flex; flex-wrap:wrap; gap:14px; margin-top:8px; }
    .stats { display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-top:18px; }
    .stat { background:#0b1222; border:1px solid #243045; border-radius:12px; padding:12px; }
    .stat-label { display:block; font-size:12px; color:#94a3b8; margin-bottom:6px; }
    .stat-value { font-size:20px; font-weight:700; }
    .chart-wrap { display:grid; gap:14px; }
    .chart-box { background:#0b1222; border:1px solid #243045; border-radius:14px; padding:10px; }
    .chart-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap; margin-bottom:8px; }
    .chart-title { font-size:14px; color:#cbd5e1; margin:0; }
    .legend { display:flex; flex-wrap:wrap; gap:8px; }
    .legend-item { display:inline-flex; align-items:center; gap:6px; background:#111827; border:1px solid #243045; border-radius:999px; padding:4px 10px; font-size:12px; color:#cbd5e1; }
    .legend-swatch { width:10px; height:10px; border-radius:999px; flex:0 0 auto; }
    .legend-value { color:#93c5fd; font-variant-numeric:tabular-nums; }
    .chart { width:100%; height:100%; }
    #priceChart { height:520px; }
    #rsiChart { height:180px; }
    #macdChart { height:220px; }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <p><a href="index.html">← Back to run index</a> · <a href="technical-indicators-report.html">Technical indicators report</a></p>
      <h1>__TITLE__</h1>
      <div class="meta">
        <span>Ticker: <strong>__TICKER__</strong></span>
        <span>Analysis date: <strong>__DATE__</strong></span>
        <span>Source vendor: <strong>Massive MCP</strong></span>
        <span>Lookback: <strong>__LOOKBACK__ calendar days</strong></span>
      </div>
      <div id="stats" class="stats"></div>
    </section>

    <section class="panel">
      <h2>Interactive chart</h2>
      <p>This view uses TradingView Lightweight Charts with Massive daily OHLCV bars. Overlays: 10 EMA, 50 SMA, 200 SMA, Bollinger Bands, and VWMA(20). When a recent company earnings anchor exists, the chart also adds an earnings-anchored AVWAP. Lower panes: RSI(14) and MACD (12,26,9).</p>
      <div class="chart-wrap">
        <div class="chart-box"><div class="chart-head"><h3 class="chart-title">Price + overlays</h3><div id="priceLegend" class="legend"></div></div><div id="priceChart" class="chart"></div></div>
        <div class="chart-box"><div class="chart-head"><h3 class="chart-title">RSI (14)</h3><div id="rsiLegend" class="legend"></div></div><div id="rsiChart" class="chart"></div></div>
        <div class="chart-box"><div class="chart-head"><h3 class="chart-title">MACD (12, 26, 9)</h3><div id="macdLegend" class="legend"></div></div><div id="macdChart" class="chart"></div></div>
      </div>
    </section>
  </main>
  <script id="chart-data" type="application/json">__PAYLOAD__</script>
  <script>
    const data = JSON.parse(document.getElementById('chart-data').textContent);
    const commonLayout = {
      layout: { background: { color: '#0b1222' }, textColor: '#d1d5db' },
      grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
      rightPriceScale: { borderColor: '#334155' },
      timeScale: { borderColor: '#334155', timeVisible: true, secondsVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    };

    function createChart(id) {
      const el = document.getElementById(id);
      return LightweightCharts.createChart(el, { ...commonLayout, width: el.clientWidth, height: el.clientHeight });
    }

    function fmt(value) {
      return value === null || value === undefined || Number.isNaN(value) ? '—' : Number(value).toFixed(2);
    }

    function addLegendItem(containerId, label, color, value) {
      const item = document.createElement('div');
      item.className = 'legend-item';
      item.innerHTML = `<span class="legend-swatch" style="background:${color}"></span><span>${label}</span><span class="legend-value">${fmt(value)}</span>`;
      document.getElementById(containerId).appendChild(item);
    }

    const hasEarningsAnchor = Boolean(data.earnings_anchor && data.earnings_anchor.anchor_label);
    const stats = [
      ['Close', data.indicator_snapshot.close],
      ['10 EMA', data.indicator_snapshot.ema10],
      ['50 SMA', data.indicator_snapshot.sma50],
      ['200 SMA', data.indicator_snapshot.sma200],
      ...(hasEarningsAnchor ? [['AVWAP from earnings', data.indicator_snapshot.avwap_from_earnings]] : []),
      ['RSI (14)', data.indicator_snapshot.rsi14],
      ['MACD', data.indicator_snapshot.macd],
      ['Signal', data.indicator_snapshot.macd_signal],
      ['Histogram', data.indicator_snapshot.macd_hist],
      ['ATR (14)', data.indicator_snapshot.atr14],
    ];
    const statsEl = document.getElementById('stats');
    if (hasEarningsAnchor) {
      const anchorCard = document.createElement('div');
      anchorCard.className = 'stat';
      anchorCard.innerHTML = `<span class="stat-label">Earnings anchor</span><span class="stat-value" style="font-size:16px">${data.earnings_anchor.anchor_label}</span>`;
      statsEl.appendChild(anchorCard);
    }
    for (const [label, value] of stats) {
      const card = document.createElement('div');
      card.className = 'stat';
      card.innerHTML = `<span class="stat-label">${label}</span><span class="stat-value">${fmt(value)}</span>`;
      statsEl.appendChild(card);
    }

    addLegendItem('priceLegend', 'Candles / Close', '#22c55e', data.indicator_snapshot.close);
    addLegendItem('priceLegend', 'EMA 10', '#38bdf8', data.indicator_snapshot.ema10);
    addLegendItem('priceLegend', 'SMA 50', '#f59e0b', data.indicator_snapshot.sma50);
    addLegendItem('priceLegend', 'SMA 200', '#f97316', data.indicator_snapshot.sma200);
    addLegendItem('priceLegend', 'Boll Upper', '#a78bfa', data.indicator_snapshot.boll_upper);
    addLegendItem('priceLegend', 'Boll Mid', '#c084fc', data.indicator_snapshot.boll_mid);
    addLegendItem('priceLegend', 'Boll Lower', '#a78bfa', data.indicator_snapshot.boll_lower);
    addLegendItem('priceLegend', 'VWMA 20', '#14b8a6', data.indicator_snapshot.vwma20);
    if (hasEarningsAnchor) {
      addLegendItem('priceLegend', 'AVWAP (earnings)', '#f472b6', data.indicator_snapshot.avwap_from_earnings);
    }
    addLegendItem('rsiLegend', 'RSI 14', '#60a5fa', data.indicator_snapshot.rsi14);
    addLegendItem('rsiLegend', 'Overbought', '#64748b', 70);
    addLegendItem('rsiLegend', 'Oversold', '#64748b', 30);
    addLegendItem('macdLegend', 'MACD', '#34d399', data.indicator_snapshot.macd);
    addLegendItem('macdLegend', 'Signal', '#f59e0b', data.indicator_snapshot.macd_signal);
    addLegendItem('macdLegend', 'Histogram', '#22c55e', data.indicator_snapshot.macd_hist);

    const priceChart = createChart('priceChart');
    const rsiChart = createChart('rsiChart');
    const macdChart = createChart('macdChart');

    const candleSeries = priceChart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444', borderVisible: false, wickUpColor: '#22c55e', wickDownColor: '#ef4444',
      lastValueVisible: false, priceLineVisible: false
    });
    candleSeries.setData(data.candles || []);

    const lineConfigs = [
      ['ema10', '#38bdf8', 2],
      ['sma50', '#f59e0b', 2],
      ['sma200', '#f97316', 2],
      ['boll_upper', '#a78bfa', 1],
      ['boll_mid', '#c084fc', 1],
      ['boll_lower', '#a78bfa', 1],
      ['vwma20', '#14b8a6', 2],
      ...(hasEarningsAnchor ? [['avwap_from_earnings', '#f472b6', 2]] : []),
    ];
    for (const [key, color, width] of lineConfigs) {
      const series = priceChart.addLineSeries({ color, lineWidth: width, lastValueVisible: false, priceLineVisible: false });
      series.setData(data[key] || []);
    }

    const rsiSeries = rsiChart.addLineSeries({ color: '#60a5fa', lineWidth: 2, lastValueVisible: false, priceLineVisible: false });
    rsiSeries.setData(data.rsi14 || []);
    const rsiUpper = rsiChart.addLineSeries({ color: '#64748b', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false });
    rsiUpper.setData((data.rsi14 || []).map(point => ({ time: point.time, value: 70 })));
    const rsiLower = rsiChart.addLineSeries({ color: '#64748b', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dashed, lastValueVisible: false, priceLineVisible: false });
    rsiLower.setData((data.rsi14 || []).map(point => ({ time: point.time, value: 30 })));

    const macdHistSeries = macdChart.addHistogramSeries({ priceFormat: { type: 'price', precision: 2, minMove: 0.01 }, lastValueVisible: false, priceLineVisible: false });
    macdHistSeries.setData(data.macd_hist || []);
    const macdSeries = macdChart.addLineSeries({ color: '#34d399', lineWidth: 2, lastValueVisible: false, priceLineVisible: false });
    macdSeries.setData(data.macd || []);
    const signalSeries = macdChart.addLineSeries({ color: '#f59e0b', lineWidth: 2, lastValueVisible: false, priceLineVisible: false });
    signalSeries.setData(data.macd_signal || []);

    priceChart.timeScale().fitContent();
    const charts = [priceChart, rsiChart, macdChart];
    let syncing = false;
    for (const chart of charts) {
      chart.timeScale().subscribeVisibleTimeRangeChange(range => {
        if (syncing || !range) return;
        syncing = true;
        for (const other of charts) {
          if (other !== chart) other.timeScale().setVisibleRange(range);
        }
        syncing = false;
      });
    }

    function resizeCharts() {
      priceChart.applyOptions({ width: document.getElementById('priceChart').clientWidth });
      rsiChart.applyOptions({ width: document.getElementById('rsiChart').clientWidth });
      macdChart.applyOptions({ width: document.getElementById('macdChart').clientWidth });
    }
    window.addEventListener('resize', resizeCharts);
  </script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", escaped_title)
        .replace("__TICKER__", escaped_ticker)
        .replace("__DATE__", escaped_date)
        .replace("__LOOKBACK__", str(TECHNICAL_CHART_LOOKBACK_DAYS))
        .replace("__PAYLOAD__", payload)
    )


def write_technical_chart_artifact(
    run_dir: Path,
    ticker: str,
    analysis_date: str,
    *,
    stack: str,
    public_base_url: Optional[str] = None,
) -> dict[str, str]:
    slug = "technical-chart"
    title = "Technical Chart"
    md_path = run_dir / f"{slug}.md"
    html_path = run_dir / f"{slug}.html"
    if md_path.exists() and html_path.exists():
        return _technical_artifact_payload(
            slug=slug,
            title=title,
            md_path=md_path,
            html_path=html_path,
            public_base_url=public_base_url,
        )
    chart_data = load_massive_chart_data(ticker, analysis_date, stack=stack)
    earnings_anchor = chart_data.get("earnings_anchor") or {}
    has_earnings_anchor = bool(earnings_anchor.get("anchor_label"))

    markdown_lines = [
        f"# {title}",
        "",
        f"- **Ticker:** `{ticker}`",
        f"- **Analysis date:** `{analysis_date}`",
        "- **Source vendor:** `massive`",
        f"- **Lookback window:** `{TECHNICAL_CHART_LOOKBACK_DAYS}` calendar days",
    ]
    if has_earnings_anchor:
        markdown_lines.append(f"- **AVWAP from earnings:** `{earnings_anchor['anchor_label']}`")
    markdown_lines.extend(
        [
            "",
            "Open `technical-chart.html` for the interactive TradingView Lightweight Charts view.",
            "",
            "Included studies:",
            "- Candles",
            "- 10 EMA",
            "- 50 SMA",
            "- 200 SMA",
            "- Bollinger Bands (20, 2)",
            "- VWMA (20)",
        ]
    )
    if has_earnings_anchor:
        markdown_lines.append("- AVWAP from the most recent earnings anchor")
    markdown_lines.extend(
        [
            "- RSI (14)",
            "- MACD (12, 26, 9)",
        ]
    )
    markdown_body = "\n".join(markdown_lines) + "\n"
    md_path.write_text(markdown_body, encoding="utf-8")
    html_path.write_text(
        render_technical_chart_html(f"{title}: {ticker} @ {analysis_date}", ticker, analysis_date, chart_data),
        encoding="utf-8",
    )
    return {
        "slug": slug,
        "title": title,
        "category": "technical",
        "markdown_path": md_path.name,
        "html_path": html_path.name,
        "markdown_url": _join_public_url(public_base_url, md_path.name) or "",
        "html_url": _join_public_url(public_base_url, html_path.name) or "",
    }


def write_technical_indicators_artifact(
    run_dir: Path,
    ticker: str,
    analysis_date: str,
    *,
    stack: str,
    public_base_url: Optional[str] = None,
    look_back_days: int = 30,
) -> dict[str, str]:
    slug = "technical-indicators-report"
    title = "Technical Indicators Report"
    md_path = run_dir / f"{slug}.md"
    html_path = run_dir / f"{slug}.html"
    if md_path.exists() and html_path.exists():
        return _technical_artifact_payload(
            slug=slug,
            title=title,
            md_path=md_path,
            html_path=html_path,
            public_base_url=public_base_url,
        )

    config = _build_stack_config(stack)
    set_config(config)
    vendor = get_vendor("technical_indicators", "get_indicators")

    sections = [
        f"# {title}",
        "",
        f"- **Ticker:** `{ticker}`",
        f"- **Analysis date:** `{analysis_date}`",
        f"- **Configured vendor:** `{vendor}`",
        f"- **Lookback window:** `{look_back_days}` calendar days",
        "",
        "This artifact is published separately from the narrative market report so the underlying technical calculations are always explicit and easy to review.",
        "",
    ]
    for indicator, label in TECHNICAL_INDICATORS:
        sections.extend(
            [
                f"## {label} (`{indicator}`)",
                "",
                "```text",
                route_to_vendor("get_indicators", ticker, indicator, analysis_date, look_back_days).strip(),
                "```",
                "",
            ]
        )

    body = "\n".join(sections).strip() + "\n"
    md_path.write_text(body, encoding="utf-8")
    html_path.write_text(
        render_markdown_html(f"{title}: {ticker} @ {analysis_date}", body),
        encoding="utf-8",
    )
    return _technical_artifact_payload(
        slug=slug,
        title=title,
        md_path=md_path,
        html_path=html_path,
        public_base_url=public_base_url,
    )


def write_report_artifacts(
    run_dir: Path,
    ticker: str,
    analysis_date: str,
    state: dict[str, Any],
    *,
    public_base_url: Optional[str] = None,
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for spec in REPORT_ARTIFACT_SPECS:
        body = spec.extractor(state)
        if not body:
            continue
        md_path = run_dir / f"{spec.slug}.md"
        html_path = run_dir / f"{spec.slug}.html"
        md_path.write_text(body + "\n", encoding="utf-8")
        html_path.write_text(
            render_markdown_html(f"{spec.title}: {ticker} @ {analysis_date}", body),
            encoding="utf-8",
        )
        artifacts.append(
            {
                "slug": spec.slug,
                "title": spec.title,
                "category": spec.category,
                "markdown_path": md_path.name,
                "html_path": html_path.name,
                "markdown_url": _join_public_url(public_base_url, md_path.name) or "",
                "html_url": _join_public_url(public_base_url, html_path.name) or "",
            }
        )
    return artifacts


def _report_body_count(state: Optional[dict[str, Any]]) -> int:
    if not state:
        return 0
    return sum(1 for spec in REPORT_ARTIFACT_SPECS if spec.extractor(state))


def _narrative_artifact_count(artifacts: list[dict[str, str]]) -> int:
    return sum(1 for artifact in artifacts if artifact.get("category") != "technical")


def write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_runtime_artifacts(
    *,
    run_dir: Path,
    metadata: dict[str, Any],
    metadata_path: Path,
    index_path: Path,
    ticker: str,
    analysis_date: str,
    state_log: Path,
    public_base_url: Optional[str] = None,
    fallback_state: Optional[dict[str, Any]] = None,
    state_log_not_before: Optional[float] = None,
) -> None:
    artifacts: list[dict[str, str]] = []
    try:
        artifacts.append(
            write_technical_indicators_artifact(
                run_dir,
                ticker,
                analysis_date,
                stack=metadata.get("stack", "fmp"),
                public_base_url=public_base_url,
            )
        )
    except Exception as exc:
        append_artifact_warning(metadata, f"technical indicators artifact failed: {exc}")
    try:
        artifacts.append(
            write_technical_chart_artifact(
                run_dir,
                ticker,
                analysis_date,
                stack=metadata.get("stack", "fmp"),
                public_base_url=public_base_url,
            )
        )
    except Exception as exc:
        append_artifact_warning(metadata, f"technical chart artifact failed: {exc}")

    state = load_state_log(state_log, min_modified_at=state_log_not_before)
    state_source = "state_log" if state else None
    if _report_body_count(fallback_state) > _report_body_count(state):
        state = fallback_state
        state_source = "stdout_state"

    if state:
        if state_source == "state_log":
            state_log_copy = run_dir / state_log.name
            state_log_copy.write_text(state_log.read_text(encoding="utf-8"), encoding="utf-8")
            metadata["state_log"] = state_log_copy.name
        else:
            state_log_copy = run_dir / f"final_state_reports_{analysis_date}.json"
            state_log_copy.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            metadata["state_log"] = state_log_copy.name
            append_artifact_warning(metadata, "used stdout state payload because state log reports were unavailable or incomplete")
        artifacts = write_report_artifacts(
            run_dir,
            ticker,
            analysis_date,
            state,
            public_base_url=public_base_url,
        ) + artifacts

    metadata["report_artifacts"] = artifacts
    metadata["report_summary"] = summarize_report_artifacts(artifacts)
    write_metadata(metadata_path, metadata)
    index_path.write_text(render_index_html(metadata), encoding="utf-8")


def render_live_html(title: str) -> str:
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <meta http-equiv=\"refresh\" content=\"15\" />
  <title>{escaped_title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{ background:#0b1020; color:#e5e7eb; font-family:Inter,system-ui,sans-serif; margin:0; }}
    header {{ padding:20px 24px; border-bottom:1px solid #223; position:sticky; top:0; background:#0b1020; z-index:10; }}
    h1 {{ margin:0 0 8px 0; font-size:20px; }}
    .meta {{ color:#9ca3af; font-size:14px; display:flex; gap:16px; flex-wrap:wrap; }}
    main {{ padding:24px; display:grid; gap:18px; max-width:1200px; margin:0 auto; }}
    a {{ color:#93c5fd; }}
    .pill {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#1f2937; color:#d1d5db; font-size:12px; }}
    .grid {{ display:grid; grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr); gap:18px; align-items:start; }}
    .panel {{ background:#111827; border:1px solid #1f2937; border-radius:14px; box-shadow:0 10px 30px rgba(0,0,0,.18); }}
    .panel h2 {{ margin:0; font-size:16px; padding:16px 18px; border-bottom:1px solid #1f2937; }}
    .panel-body {{ padding:16px 18px; }}
    .stats {{ display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }}
    .stat {{ background:#0b1222; border:1px solid #243045; border-radius:12px; padding:12px; }}
    .stat-label {{ display:block; font-size:12px; color:#94a3b8; margin-bottom:6px; }}
    .stat-value {{ font-size:22px; font-weight:700; }}
    .feed {{ display:grid; gap:12px; }}
    .entry {{ border:1px solid #22304a; border-left-width:6px; border-radius:12px; background:#0d1528; overflow:hidden; }}
    .entry-header {{ padding:10px 14px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#cbd5e1; background:rgba(255,255,255,.02); border-bottom:1px solid rgba(255,255,255,.05); }}
    .entry-body {{ padding:14px; white-space:pre-wrap; word-break:break-word; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:13px; line-height:1.55; }}
    .entry-human {{ border-left-color:#38bdf8; background:#081824; }}
    .entry-ai {{ border-left-color:#818cf8; background:#0f1530; }}
    .entry-tool {{ border-left-color:#f59e0b; background:#201506; }}
    .entry-final {{ border-left-color:#22c55e; background:#071b12; }}
    .entry-final .entry-body {{ font-family:inherit; font-size:14px; }}
    .entry-system {{ border-left-color:#64748b; background:#121826; }}
    .markdown {{ line-height:1.65; }}
    .markdown h1, .markdown h2, .markdown h3 {{ line-height:1.25; }}
    .markdown code, .markdown pre {{ background:#0b1020; border-radius:8px; }}
    .markdown pre {{ padding:16px; overflow:auto; }}
    .markdown table {{ border-collapse:collapse; width:100%; }}
    .markdown th, .markdown td {{ border:1px solid #263244; padding:8px 10px; text-align:left; vertical-align:top; }}
    .markdown blockquote {{ border-left:4px solid #374151; margin-left:0; padding-left:16px; color:#cbd5e1; }}
    .raw-log {{ max-height:70vh; overflow:auto; white-space:pre-wrap; word-break:break-word; background:#0b1222; border:1px solid #243045; border-radius:12px; padding:14px; font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size:12px; line-height:1.5; }}
    .report-groups {{ display:grid; gap:14px; margin-top:16px; }}
    .report-group {{ background:#0b1222; border:1px solid #243045; border-radius:12px; padding:12px; }}
    .report-group h3 {{ margin:0 0 8px 0; font-size:14px; }}
    .report-group ul {{ margin:0; padding-left:18px; }}
    .report-group li {{ margin:6px 0; }}
    .muted {{ color:#94a3b8; }}
    @media (max-width: 980px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div class=\"meta\">
      <span id=\"status\" class=\"pill\">Loading…</span>
      <span id=\"duration\">Duration: --</span>
      <a href=\"console.txt\">Raw text</a>
      <a href=\"index.html\">Run index</a>
      <a href=\"final.html\">Rendered final decision</a>
    </div>
  </header>
  <main>
    <div class=\"grid\">
      <section class=\"panel\">
        <h2>Live structured view</h2>
        <div class=\"panel-body\">
          <div class=\"feed\" id=\"feed\">
            <div class=\"entry entry-system\"><div class=\"entry-header\">Loading</div><div class=\"entry-body\">Waiting for output…</div></div>
          </div>
        </div>
      </section>
      <aside class=\"panel\">
        <h2>Run summary</h2>
        <div class=\"panel-body\">
          <div class=\"stats\">
            <div class=\"stat\"><span class=\"stat-label\">Human messages</span><span class=\"stat-value\" id=\"humanCount\">0</span></div>
            <div class=\"stat\"><span class=\"stat-label\">Tool calls</span><span class=\"stat-value\" id=\"toolCount\">0</span></div>
            <div class=\"stat\"><span class=\"stat-label\">AI blocks</span><span class=\"stat-value\" id=\"aiCount\">0</span></div>
            <div class=\"stat\"><span class=\"stat-label\">Final sections</span><span class=\"stat-value\" id=\"finalCount\">0</span></div>
          </div>
          <div class=\"report-groups\">
            <div>
              <strong>Published reports</strong>
              <div id=\"reportCount\" class=\"muted\">Waiting for artifacts…</div>
            </div>
            <div id=\"reportGroups\" class=\"report-groups\">
              <div class=\"report-group muted\">No reports published yet.</div>
            </div>
          </div>
          <p class=\"muted\" style=\"margin-top:16px\">Color coding: blue = human, amber = tool calls, indigo = AI output, green = final recommendation.</p>
          <details style=\"margin-top:18px\">
            <summary>Raw log</summary>
            <div id=\"log\" class=\"raw-log\">Loading output…</div>
          </details>
        </div>
      </aside>
    </div>
  </main>
  <script>
    const logEl = document.getElementById('log');
    const feedEl = document.getElementById('feed');
    const statusEl = document.getElementById('status');
    const durationEl = document.getElementById('duration');
    const humanCountEl = document.getElementById('humanCount');
    const toolCountEl = document.getElementById('toolCount');
    const aiCountEl = document.getElementById('aiCount');
    const finalCountEl = document.getElementById('finalCount');
    const reportCountEl = document.getElementById('reportCount');
    const reportGroupsEl = document.getElementById('reportGroups');

    function escapeHtml(value) {{
      return value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    }}

    function renderInlineMarkdown(md) {{
      let body = escapeHtml(md.trim());
      body = body.replace(/^###\\s+(.*)$/gm, '<h3>$1</h3>');
      body = body.replace(/^##\\s+(.*)$/gm, '<h2>$1</h2>');
      body = body.replace(/^#\\s+(.*)$/gm, '<h1>$1</h1>');
      body = body.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
      body = body.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
      body = body.replace(/`([^`]+)`/g, '<code>$1</code>');
      body = body.replace(/\n\n/g, '</p><p>');
      body = body.replace(/\n-\\s+/g, '\n• ');
      return '<div class="markdown"><p>' + body + '</p></div>';
    }}

    function buildEntry(kind, label, body, isMarkdown=false) {{
      const wrapper = document.createElement('div');
      wrapper.className = 'entry entry-' + kind;
      const header = document.createElement('div');
      header.className = 'entry-header';
      header.textContent = label;
      const content = document.createElement('div');
      content.className = 'entry-body';
      if (isMarkdown) {{
        content.innerHTML = renderInlineMarkdown(body);
      }} else {{
        content.textContent = body.trim() || '…';
      }}
      wrapper.appendChild(header);
      wrapper.appendChild(content);
      return wrapper;
    }}

    function parseStructuredEntries(logText) {{
      const entries = [];
      const finalMatches = [...logText.matchAll(/=== FINAL_DECISION_MARKDOWN_BEGIN ===([\\s\\S]*?)=== FINAL_DECISION_MARKDOWN_END ===/g)];
      const cleaned = logText
        .replace(/=== FINAL_DECISION_MARKDOWN_BEGIN ===[\\s\\S]*?=== FINAL_DECISION_MARKDOWN_END ===/g, '')
        .replace(/=== FINAL_STATE_REPORTS_JSON_BEGIN ===[\\s\\S]*?=== FINAL_STATE_REPORTS_JSON_END ===/g, '')
        .trim();
      const humanHeader = '================================ Human Message =================================';
      const aiHeader = '================================== Ai Message ==================================';
      const lines = cleaned.split('\n');
      let current = [];
      let currentKind = 'system';
      let currentLabel = 'System / setup';

      function flushCurrent() {{
        const body = current.join('\n').trim();
        if (body && !(currentKind === 'human' && body === 'Continue')) {{
          entries.push({{ kind: currentKind, label: currentLabel, body }});
        }}
        current = [];
      }}

      for (const line of lines) {{
        if (line === humanHeader) {{
          flushCurrent();
          currentKind = 'human';
          currentLabel = 'Human message';
          continue;
        }}
        if (line === aiHeader) {{
          flushCurrent();
          currentKind = 'ai';
          currentLabel = 'AI output';
          continue;
        }}
        current.push(line);
      }}
      flushCurrent();

      const expanded = [];
      for (const entry of entries) {{
        if (entry.kind !== 'ai') {{
          expanded.push(entry);
          continue;
        }}
        const toolRegex = /<tool_call>[\\s\\S]*?(?:<\\/tool_call>|$)/g;
        const toolMatches = [...entry.body.matchAll(toolRegex)];
        const plain = entry.body.replace(toolRegex, '').trim();
        if (plain) expanded.push({{ kind:'ai', label:'AI output', body: plain }});
        for (const match of toolMatches) {{
          expanded.push({{ kind:'tool', label:'Tool call', body: match[0] }});
        }}
      }}

      for (const match of finalMatches) {{
        expanded.push({{ kind:'final', label:'Final recommendation', body: match[1].trim(), markdown: true }});
      }}
      return expanded;
    }}

    function renderEntries(logText) {{
      const entries = parseStructuredEntries(logText);
      feedEl.replaceChildren();
      if (!entries.length) {{
        feedEl.appendChild(buildEntry('system', 'No output yet', 'Waiting for output…'));
      }}
      let human = 0, tool = 0, ai = 0, final = 0;
      for (const entry of entries) {{
        if (entry.kind === 'human') human += 1;
        if (entry.kind === 'tool') tool += 1;
        if (entry.kind === 'ai') ai += 1;
        if (entry.kind === 'final') final += 1;
        feedEl.appendChild(buildEntry(entry.kind, entry.label, entry.body, Boolean(entry.markdown)));
      }}
      humanCountEl.textContent = String(human);
      toolCountEl.textContent = String(tool);
      aiCountEl.textContent = String(ai);
      finalCountEl.textContent = String(final);
    }}

    function renderReportArtifacts(meta) {{
      const summary = meta.report_summary || {{}};
      const groups = summary.groups || [];
      const artifacts = meta.report_artifacts || [];
      reportCountEl.textContent = `${{summary.count || artifacts.length || 0}} reports published.`;
      reportGroupsEl.replaceChildren();
      if (!groups.length) {{
        const empty = document.createElement('div');
        empty.className = 'report-group muted';
        empty.textContent = 'No reports published yet.';
        reportGroupsEl.appendChild(empty);
        return;
      }}
      for (const group of groups) {{
        const card = document.createElement('section');
        card.className = 'report-group';
        const heading = document.createElement('h3');
        heading.textContent = group.title || 'Reports';
        card.appendChild(heading);
        const list = document.createElement('ul');
        for (const artifact of group.artifacts || []) {{
          const item = document.createElement('li');
          const htmlHref = artifact.html_url || artifact.html_path || '#';
          const mdHref = artifact.markdown_url || artifact.markdown_path || '#';
          item.innerHTML = `<strong>${{escapeHtml(artifact.title || artifact.slug || 'Report')}}</strong>: <a href="${{htmlHref}}">HTML</a> · <a href="${{mdHref}}">Markdown</a>`;
          list.appendChild(item);
        }}
        card.appendChild(list);
        reportGroupsEl.appendChild(card);
      }}
    }}

    async function refresh() {{
      try {{
        const [logResp, metaResp] = await Promise.all([
          fetch('console.txt?ts=' + Date.now()),
          fetch('metadata.json?ts=' + Date.now()),
        ]);
        const logText = await logResp.text();
        logEl.textContent = logText;
        renderEntries(logText);
        const meta = await metaResp.json();
        statusEl.textContent = meta.status || 'unknown';
        durationEl.textContent = 'Duration: ' + (meta.duration_hms || '--');
        renderReportArtifacts(meta);
        window.scrollTo(0, document.body.scrollHeight);
      }} catch (err) {{
        statusEl.textContent = 'refresh failed';
      }}
    }}
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>
"""


def render_index_html(metadata: dict) -> str:
    title = html.escape(metadata.get("title", "Ticker Agents Run"))
    report_artifacts = metadata.get("report_artifacts", []) or []
    report_summary_meta = metadata.get("report_summary") or summarize_report_artifacts(report_artifacts)
    links = [
        ("Live HTML", metadata.get("live_url") or "live.html"),
        ("Raw console text", metadata.get("raw_url") or "console.txt"),
        ("Run metadata", "metadata.json"),
    ]
    if metadata.get("state_log"):
        links.append(("Full state log", metadata["state_log"]))
    if metadata.get("has_final_markdown"):
        links.append(("Rendered final decision", metadata.get("final_html_url") or "final.html"))
        links.append(("Final decision markdown", metadata.get("final_md_url") or "final.md"))
    for artifact in report_artifacts:
        links.append((f"{artifact['title']} (HTML)", artifact.get("html_url") or artifact["html_path"]))
        links.append((f"{artifact['title']} (Markdown)", artifact.get("markdown_url") or artifact["markdown_path"]))
    link_html = "".join(f'<li><a href="{href}">{label}</a></li>' for label, href in links)
    report_summary = ""
    if report_artifacts:
        total_reports = report_summary_meta.get("count", len(report_artifacts))
        grouped_blocks = []
        for group in report_summary_meta.get("groups", []):
            items = "".join(
                (
                    f'<li><strong>{html.escape(artifact["title"])}</strong>: '
                    f'<a href="{artifact.get("html_url") or artifact["html_path"]}">HTML</a> · '
                    f'<a href="{artifact.get("markdown_url") or artifact["markdown_path"]}">Markdown</a></li>'
                )
                for artifact in group.get("artifacts", [])
            )
            grouped_blocks.append(
                f'<section><h3>{html.escape(group.get("title", "Reports"))}</h3><ul>{items}</ul></section>'
            )
        report_summary = (
            f"<h2>Published reports</h2>"
            f"<p>{total_reports} reports published.</p>"
            + "".join(grouped_blocks)
        )
    details = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in [
            ("ticker", metadata.get("ticker")),
            ("analysis_date", metadata.get("analysis_date")),
            ("status", metadata.get("status")),
            ("started_at", format_timestamp_for_display(metadata.get("started_at"))),
            ("finished_at", format_timestamp_for_display(metadata.get("finished_at"))),
            ("duration", metadata.get("duration_hms") or "—"),
            ("exit_code", metadata.get("exit_code")),
            ("artifact_warning", metadata.get("artifact_warning") or "—"),
            ("command", metadata.get("command")),
        ]
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ background:#0b1020; color:#e5e7eb; font-family:Inter,system-ui,sans-serif; padding:24px; }}
    a {{ color:#93c5fd; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 1100px; }}
    th, td {{ border: 1px solid #1f2937; padding: 8px 12px; text-align: left; vertical-align: top; }}
    th {{ width: 180px; background:#111827; }}
    .box {{ background:#111827; border:1px solid #1f2937; border-radius:12px; padding:20px; max-width:1100px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class=\"box\">
    <p>This run stores both CLI-friendly text and browser-friendly HTML output.</p>
    <ul>{link_html}</ul>
    {report_summary}
    <table>{details}</table>
  </div>
</body>
</html>
"""


def render_markdown_html(title: str, body: str) -> str:
    markdown_warning = ""
    try:
        import markdown

        rendered = markdown.markdown(body, extensions=["fenced_code", "tables", "sane_lists"])
    except ModuleNotFoundError:
        markdown_warning = (
            "<p><strong>Note:</strong> Python-Markdown was unavailable, so this final output is shown "
            "in a plain-text fallback view.</p>"
        )
        rendered = f"<pre>{html.escape(body)}</pre>"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin:0; background:#0b1020; color:#e5e7eb; font-family:Inter,system-ui,sans-serif; }}
    article {{ max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; line-height: 1.65; }}
    h1,h2,h3 {{ line-height:1.25; }}
    code, pre {{ background:#111827; border-radius:8px; }}
    pre {{ padding:16px; overflow:auto; }}
    table {{ border-collapse: collapse; width:100%; }}
    th, td {{ border:1px solid #1f2937; padding:8px 10px; text-align:left; vertical-align:top; }}
    a {{ color:#93c5fd; }}
    blockquote {{ border-left:4px solid #374151; margin-left:0; padding-left:16px; color:#cbd5e1; }}
  </style>
</head>
<body>
  <article>
    <p><a href=\"index.html\">← Back to run index</a></p>
    {markdown_warning}
    {rendered}
  </article>
</body>
</html>
"""


def ensure_http_server(output_root: Path, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            return
    subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--directory", str(output_root)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.5)


def stream_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    out_handle,
    capture: list[str],
    on_output: Optional[Callable[[], None]] = None,
) -> int:
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        out_handle.write(line)
        out_handle.flush()
        capture.append(line)
        if on_output:
            on_output()
    exit_code = process.wait()
    if on_output:
        on_output()
    return exit_code


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ticker agents with captured output.")
    parser.add_argument("ticker", nargs="?", help="Ticker symbol, defaults to SPY")
    parser.add_argument("analysis_date", nargs="?", help="Analysis date YYYY-MM-DD, defaults to today")
    parser.add_argument(
        "--stack",
        choices=sorted(STACK_EXAMPLES),
        default=os.getenv("TICKER_AGENTS_STACK", "fmp"),
        help="Example stack to run: fmp (default, Massive+FMP hybrid) or grounded",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port for the output viewer")
    parser.add_argument(
        "--public-host",
        default=DEFAULT_PUBLIC_HOST,
        help=(
            "Host/IP to print for browser URLs. Defaults to TICKER_AGENTS_PUBLIC_HOST "
            "or the current machine's detected LAN IP."
        ),
    )
    return parser.parse_args(argv)


def run(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ticker, analysis_date = resolve_run_request(args.ticker, args.analysis_date)

    def _handle_signal(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    repo_root = Path(__file__).resolve().parents[1]
    output_root = repo_root / "output"
    output_root.mkdir(exist_ok=True)

    started_at = datetime.now()
    slug = build_run_slug(ticker, analysis_date, started_at)
    paths = build_paths(output_root / slug)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    ensure_http_server(output_root, args.port)

    public_host = detect_public_host(args.public_host)
    display_live_url = public_run_url(args.port, slug, "live.html", public_host)
    display_index_url = public_run_url(args.port, slug, "index.html", public_host)
    display_raw_url = public_run_url(args.port, slug, "console.txt", public_host)
    display_final_url = public_run_url(args.port, slug, "final.html", public_host)
    live_url = "live.html"
    index_url = "index.html"
    raw_url = "console.txt"

    metadata = {
        "title": f"Ticker Agents Run: {ticker} @ {analysis_date}",
        "ticker": ticker,
        "analysis_date": analysis_date,
        "stack": args.stack,
        "status": "running",
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": None,
        "duration_hms": "00:00:00",
        "exit_code": None,
        "slug": slug,
        "command": None,
        "live_url": live_url,
        "index_url": index_url,
        "raw_url": raw_url,
        "final_html_url": "final.html",
        "final_md_url": "final.md",
        "has_final_markdown": False,
        "report_artifacts": [],
        "report_summary": {"count": 0, "groups": []},
        "state_log": None,
        "artifact_warning": None,
    }
    write_metadata(paths.metadata_json, metadata)
    paths.live_html.write_text(render_live_html(metadata["title"]), encoding="utf-8")
    paths.index_html.write_text(render_index_html(metadata), encoding="utf-8")

    banner = (
        f"Run directory: {paths.run_dir}\n"
        f"Live HTML: {display_live_url}\n"
        f"Index: {display_index_url}\n"
        f"Raw text: {display_raw_url}\n\n"
    )
    print(banner, end="")

    combined_output: list[str] = []
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "dummy")
    state_log = state_log_path(ticker, analysis_date)
    if not state_log.exists():
        try:
            state_log = state_log_path_from_config(ticker, analysis_date, _build_stack_config(args.stack))
        except Exception as exc:
            append_artifact_warning(metadata, f"could not resolve stack state log path: {exc}")
    last_runtime_publish_at = 0.0
    runtime_publish_warning: Optional[str] = None

    def maybe_publish_runtime_artifacts(force: bool = False) -> None:
        nonlocal last_runtime_publish_at, runtime_publish_warning
        now = time.monotonic()
        if not force and now - last_runtime_publish_at < 2:
            return
        try:
            publish_runtime_artifacts(
                run_dir=paths.run_dir,
                metadata=metadata,
                metadata_path=paths.metadata_json,
                index_path=paths.index_html,
                ticker=ticker,
                analysis_date=analysis_date,
                state_log=state_log,
                public_base_url=None,
                fallback_state=extract_state_reports("".join(combined_output)),
                state_log_not_before=started_at.timestamp() - 1.0,
            )
        except Exception as exc:
            warning = f"runtime report publish failed: {exc}"
            if runtime_publish_warning != warning:
                append_artifact_warning(metadata, warning)
                runtime_publish_warning = warning
        last_runtime_publish_at = now

    maybe_publish_runtime_artifacts(force=True)

    try:
        with paths.console_txt.open("w", encoding="utf-8") as out_handle:
            header = (
                f"# Ticker Agents Run\n"
                f"ticker: {ticker}\n"
                f"analysis_date: {analysis_date}\n"
                f"started_at: {metadata['started_at']}\n"
                f"live_html: {display_live_url}\n"
                f"raw_text: {display_raw_url}\n\n"
            )
            out_handle.write(banner)
            out_handle.write(header)
            out_handle.flush()
            combined_output.extend([banner, header])

            sync_exit = stream_command(["uv", "sync"], repo_root, env, out_handle, combined_output)
            if sync_exit != 0:
                metadata["status"] = "failed"
                metadata["exit_code"] = sync_exit
            else:
                app_command = build_app_command(repo_root, ticker, analysis_date, args.stack)
                metadata["command"] = " ".join(app_command)
                write_metadata(paths.metadata_json, metadata)
                app_exit = stream_command(
                    app_command,
                    repo_root,
                    env,
                    out_handle,
                    combined_output,
                    on_output=maybe_publish_runtime_artifacts,
                )
                metadata["status"] = "completed" if app_exit == 0 else "failed"
                metadata["exit_code"] = app_exit
    except KeyboardInterrupt as exc:
        metadata["status"] = "interrupted"
        metadata["exit_code"] = 130
        interruption_note = f"\n\n# Interrupted\n{exc}\n"
        combined_output.append(interruption_note)
        with paths.console_txt.open("a", encoding="utf-8") as out_handle:
            out_handle.write(interruption_note)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)

    # Recompute elapsed correctly outside file handle to include whole run.
    finished_at = datetime.now()
    duration_seconds = (finished_at - started_at).total_seconds()
    metadata["finished_at"] = finished_at.isoformat(timespec="seconds")
    metadata["duration_hms"] = format_duration(duration_seconds)

    full_text = "".join(combined_output)
    final_md = extract_final_decision(full_text)
    final_state_reports = extract_state_reports(full_text)
    maybe_publish_runtime_artifacts(force=True)
    try:
        if final_md:
            paths.final_md.write_text(final_md + "\n", encoding="utf-8")
            paths.final_html.write_text(
                render_markdown_html(f"Final Decision: {ticker} @ {analysis_date}", final_md),
                encoding="utf-8",
            )
            metadata["has_final_markdown"] = True
        else:
            metadata["has_final_markdown"] = False

        if (
            final_state_reports
            and _narrative_artifact_count(metadata.get("report_artifacts", []))
            < _report_body_count(final_state_reports)
        ):
            publish_runtime_artifacts(
                run_dir=paths.run_dir,
                metadata=metadata,
                metadata_path=paths.metadata_json,
                index_path=paths.index_html,
                ticker=ticker,
                analysis_date=analysis_date,
                state_log=state_log,
                public_base_url=None,
                fallback_state=final_state_reports,
                state_log_not_before=started_at.timestamp() - 1.0,
            )
        maybe_publish_runtime_artifacts(force=True)
    except Exception as exc:
        metadata["has_final_markdown"] = paths.final_md.exists()
        append_artifact_warning(metadata, f"final artifact rendering failed: {exc}")
        if metadata["status"] == "completed":
            metadata["status"] = "completed_with_warnings"
    finally:
        if metadata.get("artifact_warning") and metadata["status"] == "completed":
            metadata["status"] = "completed_with_warnings"
        with paths.console_txt.open("a", encoding="utf-8") as out_handle:
            out_handle.write(f"\n\n# Run summary\n")
            out_handle.write(f"status: {metadata['status']}\n")
            out_handle.write(f"finished_at: {metadata['finished_at']}\n")
            out_handle.write(f"duration: {metadata['duration_hms']}\n")
            out_handle.write(f"exit_code: {metadata['exit_code']}\n")
            out_handle.write(f"report_artifacts: {len(metadata.get('report_artifacts', []))}\n")
            if metadata.get("artifact_warning"):
                out_handle.write(f"artifact_warning: {metadata['artifact_warning']}\n")

        write_metadata(paths.metadata_json, metadata)
        paths.index_html.write_text(render_index_html(metadata), encoding="utf-8")

    print()
    print(f"Completed with status={metadata['status']} in {metadata['duration_hms']}")
    print(f"Live HTML: {display_live_url}")
    print(f"Index: {display_index_url}")
    print(f"Raw text: {display_raw_url}")
    if metadata["has_final_markdown"]:
        print(f"Rendered final decision: {display_final_url}")
    return int(metadata["exit_code"] or 0)


if __name__ == "__main__":
    raise SystemExit(run())
