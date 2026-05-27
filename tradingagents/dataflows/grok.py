from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from .config import get_config
from .mcp_support import call_tool, news_mcp_url

_XAI_API_KEY = os.getenv("XAI_API_KEY")
_XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
_XAI_MODEL = os.getenv("XAI_MODEL", "grok-4.3")


def _headers() -> Dict[str, str]:
    if not _XAI_API_KEY:
        return {}
    return {
        "Authorization": f"Bearer {_XAI_API_KEY}",
        "Content-Type": "application/json",
    }


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _XAI_API_KEY:
        raise RuntimeError("XAI_API_KEY is not set")
    response = requests.post(
        f"{_XAI_BASE_URL}{path}",
        headers=_headers(),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _header(title: str) -> str:
    return f"# {title}\n# Data source: Grok / xAI\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"


def _extract_output_text(payload: Dict[str, Any]) -> str:
    if payload.get("output_text"):
        return str(payload["output_text"]).strip()

    collected: List[str] = []
    for item in payload.get("output") or []:
        for content in item.get("content") or []:
            text = content.get("text")
            if text:
                collected.append(text)
    if collected:
        return "\n".join(collected).strip()

    choices = payload.get("choices") or []
    if choices:
        return ((choices[0].get("message") or {}).get("content") or "").strip()
    return ""


def _append_citations(text: str, payload: Dict[str, Any]) -> str:
    citations = payload.get("citations") or []
    if not citations:
        return text
    lines = [text, "", "Citations:"] if text else ["Citations:"]
    for citation in citations:
        if isinstance(citation, dict):
            title = citation.get("title") or citation.get("url") or str(citation)
            url = citation.get("url")
            lines.append(f"- {title}" + (f" ({url})" if url and url != title else ""))
        else:
            lines.append(f"- {citation}")
    return "\n".join(lines).strip()


def _responses_prompt(user_prompt: str, tools: List[Dict[str, Any]]) -> str:
    payload = {
        "model": _XAI_MODEL,
        "input": [{"role": "user", "content": user_prompt}],
        "tools": tools,
    }
    response = _post("/responses", payload)
    return _append_citations(_extract_output_text(response), response)


def _run_grok_via_mcp(
    prompt: str,
    *,
    use_web_search: Optional[bool] = None,
    use_x_search: Optional[bool] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> str:
    mcp_url = news_mcp_url()
    if not mcp_url:
        raise RuntimeError("XAI_API_KEY is not set and news_mcp_url is not configured")
    payload: Dict[str, Any] = {"prompt": prompt}
    if use_web_search is not None:
        payload["use_web_search"] = use_web_search
    if use_x_search is not None:
        payload["use_x_search"] = use_x_search
    if from_date:
        payload["from_date"] = from_date
    if to_date:
        payload["to_date"] = to_date
    result = call_tool(mcp_url, "run_grok_prompt", payload)
    return (result.get("text") or result.get("result") or "").strip()


def get_news(ticker: str, start_date: str, end_date: str):
    try:
        prompt = (
            f"Using live web search, summarize the most important news for {ticker} between {start_date} and {end_date}. "
            "Return a concise markdown briefing with bullet points for catalysts, risks, and notable headlines. "
            "Only include information you can ground in retrieved sources."
        )
        text = _responses_prompt(prompt, [{"type": "web_search"}]) if _XAI_API_KEY else _run_grok_via_mcp(prompt, use_web_search=True)
        if not text:
            return f"No Grok news found for {ticker} between {start_date} and {end_date}"
        return _header(f"{ticker.upper()} News, from {start_date} to {end_date}") + text + "\n"
    except Exception as e:
        return f"Error retrieving Grok news for {ticker}: {e}"


def get_global_news(curr_date: str, look_back_days: Optional[int] = None, limit: Optional[int] = None):
    try:
        config = get_config()
        look_back_days = look_back_days or config["global_news_lookback_days"]
        start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
        prompt = (
            f"Using live web search, summarize the most important macro and market news between {start_date} and {curr_date}. "
            "Cover central banks, inflation, earnings backdrop, geopolitics, commodities, and risk sentiment. "
            "Return markdown bullets grouped by theme."
        )
        text = _responses_prompt(prompt, [{"type": "web_search"}]) if _XAI_API_KEY else _run_grok_via_mcp(prompt, use_web_search=True)
        if not text:
            return f"No global news found for {curr_date}"
        return _header(f"Global Market News, from {start_date} to {curr_date}") + text + "\n"
    except Exception as e:
        return f"Error retrieving Grok global news: {e}"


def get_x_sentiment_report(ticker: str, start_date: str, end_date: str):
    try:
        prompt = (
            f"Use X Search to analyze sentiment on X about {ticker} from {start_date} to {end_date}. "
            "Return markdown with: overall sentiment, bullish themes, bearish themes, whether sentiment looks crowded, and a short evidence section. "
            "Do not invent posts; rely only on the retrieved X data."
        )
        text = _responses_prompt(prompt, [{"type": "x_search"}]) if _XAI_API_KEY else _run_grok_via_mcp(
            prompt,
            use_x_search=True,
            from_date=start_date,
            to_date=end_date,
        )
        if not text:
            return f"No X sentiment data found for {ticker} between {start_date} and {end_date}"
        return _header(f"X Sentiment for {ticker.upper()}, from {start_date} to {end_date}") + text + "\n"
    except Exception as e:
        return f"Error retrieving Grok X sentiment for {ticker}: {e}"
