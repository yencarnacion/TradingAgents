from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import requests

from .config import get_config
from .mcp_support import call_tool, news_mcp_url

_PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
_PERPLEXITY_BASE_URL = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")


def _headers() -> Dict[str, str]:
    if not _PERPLEXITY_API_KEY:
        return {}
    return {
        "Authorization": f"Bearer {_PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not _PERPLEXITY_API_KEY:
        raise RuntimeError("PERPLEXITY_API_KEY is not set")
    response = requests.post(
        f"{_PERPLEXITY_BASE_URL}{path}",
        headers=_headers(),
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _header(title: str) -> str:
    return f"# {title}\n# Data source: Perplexity\n# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"


def _format_search_results(title: str, results: Iterable[Dict[str, Any]]) -> str:
    lines = [_header(title).rstrip(), ""]
    count = 0
    for item in results:
        lines.append(f"### {item.get('title', 'Untitled')}")
        if item.get("date"):
            lines.append(f"Published: {item['date']}")
        if item.get("snippet"):
            lines.append(item["snippet"])
        if item.get("url"):
            lines.append(f"Link: {item['url']}")
        lines.append("")
        count += 1
    if count == 0:
        return f"{_header(title)}No search results returned."
    return "\n".join(lines).strip() + "\n"


def _search(query: str, max_results: int, **extra: Any) -> List[Dict[str, Any]]:
    payload = {"query": query, "max_results": max_results}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return (_post("/search", payload).get("results") or [])


def _sonar_summary(prompt: str, search_mode: str = "web", domains: Optional[List[str]] = None) -> str:
    payload: Dict[str, Any] = {
        "model": os.getenv("PERPLEXITY_SONAR_MODEL", "sonar-pro"),
        "messages": [
            {"role": "system", "content": "You are a grounded financial research assistant. Cite only information found via search."},
            {"role": "user", "content": prompt},
        ],
        "search_mode": search_mode,
        "temperature": 0.1,
    }
    if domains:
        payload["search_domain_filter"] = domains
    response = _post("/v1/sonar", payload)
    choices = response.get("choices") or []
    content = ""
    if choices:
        content = ((choices[0].get("message") or {}).get("content") or "").strip()
    citations = response.get("citations") or []
    citation_lines = [f"- {c}" for c in citations if c]
    if citation_lines:
        content += "\n\nCitations:\n" + "\n".join(citation_lines)
    return content.strip()


def _run_perplexity_via_mcp(query: str, *, model: Optional[str] = None, search_mode: Optional[str] = None) -> str:
    mcp_url = news_mcp_url()
    if not mcp_url:
        raise RuntimeError("PERPLEXITY_API_KEY is not set and news_mcp_url is not configured")
    payload = {"query": query}
    if model:
        payload["model"] = model
    if search_mode:
        payload["search_mode"] = search_mode
    result = call_tool(mcp_url, "run_perplexity_query", payload)
    return (result.get("text") or result.get("result") or "").strip()


def get_news(ticker: str, start_date: str, end_date: str):
    try:
        if _PERPLEXITY_API_KEY:
            article_limit = get_config()["news_article_limit"]
            query = f"{ticker} latest company news earnings guidance SEC filing catalyst"
            results = _search(
                query,
                article_limit,
                search_after_date_filter=datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y"),
                search_before_date_filter=datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y"),
                search_domain_filter=["sec.gov", "investorrelations", "bloomberg.com", "reuters.com", "wsj.com", "finance.yahoo.com"],
            )
            search_block = _format_search_results(f"{ticker.upper()} News, from {start_date} to {end_date}", results)
            sec_prompt = (
                f"Summarize the most relevant SEC filings and official-company disclosures for {ticker} "
                f"between {start_date} and {end_date}. Focus on catalysts, risks, guidance changes, and items a trading agent should care about."
            )
            sec_summary = _sonar_summary(sec_prompt, search_mode="sec", domains=["sec.gov"]) if _PERPLEXITY_API_KEY else ""
            if sec_summary:
                search_block += "\n## SEC-grounded summary\n\n" + sec_summary + "\n"
            return search_block

        prompt = (
            f"Summarize the most important company-specific news, earnings/guidance changes, catalysts, and SEC disclosures for {ticker} "
            f"between {start_date} and {end_date}. Return markdown with bullets and a short SEC-grounded section if relevant."
        )
        text = _run_perplexity_via_mcp(prompt, search_mode="web")
        if not text:
            return f"No Perplexity news found for {ticker} between {start_date} and {end_date}"
        return _header(f"{ticker.upper()} News, from {start_date} to {end_date}") + text + "\n"
    except Exception as e:
        return f"Error retrieving Perplexity news for {ticker}: {e}"


def get_global_news(curr_date: str, look_back_days: Optional[int] = None, limit: Optional[int] = None):
    try:
        config = get_config()
        look_back_days = look_back_days or config["global_news_lookback_days"]
        limit = limit or config["global_news_article_limit"]
        start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days)

        if _PERPLEXITY_API_KEY:
            queries = config["global_news_queries"]
            start_date = start_dt.strftime("%m/%d/%Y")
            end_date = datetime.strptime(curr_date, "%Y-%m-%d").strftime("%m/%d/%Y")

            deduped: Dict[str, Dict[str, Any]] = {}
            per_query = max(1, min(20, limit))
            for query in queries:
                for item in _search(
                    query,
                    per_query,
                    search_after_date_filter=start_date,
                    search_before_date_filter=end_date,
                ):
                    key = item.get("url") or item.get("title") or str(len(deduped))
                    deduped.setdefault(key, item)
                    if len(deduped) >= limit:
                        break
                if len(deduped) >= limit:
                    break

            if not deduped:
                return f"No global news found for {curr_date}"

            return _format_search_results(
                f"Global Market News, from {start_dt.strftime('%Y-%m-%d')} to {curr_date}",
                list(deduped.values())[:limit],
            )

        prompt = (
            f"Summarize the most important macro and market news between {start_dt.strftime('%Y-%m-%d')} and {curr_date}. "
            "Cover central banks, inflation, earnings backdrop, geopolitics, commodities, and broad risk sentiment. Return markdown bullets grouped by theme."
        )
        text = _run_perplexity_via_mcp(prompt, search_mode="web")
        if not text:
            return f"No global news found for {curr_date}"
        return _header(f"Global Market News, from {start_dt.strftime('%Y-%m-%d')} to {curr_date}") + text + "\n"
    except Exception as e:
        return f"Error retrieving Perplexity global news: {e}"
