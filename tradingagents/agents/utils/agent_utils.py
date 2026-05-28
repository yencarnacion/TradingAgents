from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timedelta
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.market_context_tools import (
    get_intraday_bars,
    get_session_bars,
    get_ticker_snapshot,
    get_market_regime,
    get_last_trade,
    get_nbbo_quotes,
    get_options_chain,
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str, asset_type: str = "stock") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    instrument_label = "asset" if asset_type == "crypto" else "instrument"
    extra_hint = (
        " Treat it as a crypto asset rather than a company, and do not assume company fundamentals are available."
        if asset_type == "crypto"
        else ""
    )
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_INVOKE_RE = re.compile(r"<invoke\s+name=['\"]([^'\"]+)['\"]\s*>\s*(.*?)\s*</invoke>", re.DOTALL)
_NAMED_PARAMETER_RE = re.compile(
    r"<parameter\s+name=['\"]([^'\"]+)['\"]\s*>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_FENCED_CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```", re.DOTALL)
_SIMPLE_CALL_RE = re.compile(
    r"^\s*(?:call_)?([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$",
    re.DOTALL,
)
_CALL_ARG_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"[^\"]*\"|'[^']*'|[^,]+)"
)


def _coerce_tool_parameter(value: str):
    text = value.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"[-+]?\d*\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _tool_call_from_mapping(payload: dict[str, Any], call_id: int) -> dict[str, Any] | None:
    action = payload.get("action")
    if isinstance(action, dict):
        action_call = _tool_call_from_mapping(action, call_id)
        if action_call is not None:
            return action_call

    function = payload.get("function")
    if isinstance(function, dict):
        function_name = function.get("name")
        raw_args = function.get("arguments") or payload.get("args") or payload.get("parameters") or {}
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = {}
    else:
        function_name = payload.get("tool") or payload.get("name") or function
        raw_args = payload.get("args") or payload.get("parameters") or {}

    if not isinstance(function_name, str) or not function_name.strip():
        return None
    if not isinstance(raw_args, dict):
        raw_args = {}

    return {
        "name": function_name.strip().removeprefix("call_"),
        "args": raw_args,
        "id": f"call_{call_id}",
        "type": "tool_call",
    }


def _extract_json_tool_calls(content: str, start_index: int) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    candidates = [match.group(1).strip() for match in _FENCED_CODE_RE.finditer(content)]
    stripped = content.strip()
    if stripped.startswith(("{", "[")):
        candidates.append(stripped)

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            tool_call = _tool_call_from_mapping(item, start_index + len(calls))
            if tool_call is not None:
                calls.append(tool_call)

    return calls


def _extract_simple_call_tool_calls(content: str, start_index: int) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    candidates = [match.group(1).strip() for match in _FENCED_CODE_RE.finditer(content)]
    stripped = content.strip()
    if _SIMPLE_CALL_RE.match(stripped):
        candidates.append(stripped)

    for candidate in candidates:
        call_match = _SIMPLE_CALL_RE.match(candidate)
        if not call_match:
            continue

        function_name = call_match.group(1).strip().removeprefix("call_")
        args_text = call_match.group(2)
        args: dict[str, Any] = {}
        for key, raw_value in _CALL_ARG_RE.findall(args_text):
            value = raw_value.strip()
            try:
                args[key] = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                args[key] = _coerce_tool_parameter(value)

        calls.append(
            {
                "name": function_name,
                "args": args,
                "id": f"call_{start_index + len(calls)}",
                "type": "tool_call",
            }
        )

    return calls


def extract_tool_calls_from_markup(content: str) -> list[dict]:
    """Parse DeepSeek/Qwen-style XML-ish tool-call markup into LangChain tool_calls."""
    if not isinstance(content, str):
        return []

    tool_calls: list[dict] = []
    for block_match in _TOOL_CALL_BLOCK_RE.finditer(content):
        block = block_match.group(1)
        function_match = _FUNCTION_RE.search(block)
        if not function_match:
            continue
        function_name = function_match.group(1).strip()
        body = function_match.group(2)
        args = {
            name.strip(): _coerce_tool_parameter(value)
            for name, value in _PARAMETER_RE.findall(body)
        }
        tool_calls.append(
            {
                "name": function_name,
                "args": args,
                "id": f"call_{len(tool_calls) + 1}",
                "type": "tool_call",
            }
        )

    for invoke_match in _INVOKE_RE.finditer(content):
        function_name = invoke_match.group(1).strip()
        body = invoke_match.group(2)
        args = {
            name.strip(): _coerce_tool_parameter(value)
            for name, value in _NAMED_PARAMETER_RE.findall(body)
        }
        tool_calls.append(
            {
                "name": function_name,
                "args": args,
                "id": f"call_{len(tool_calls) + 1}",
                "type": "tool_call",
            }
        )
    tool_calls.extend(_extract_json_tool_calls(content, len(tool_calls) + 1))
    tool_calls.extend(_extract_simple_call_tool_calls(content, len(tool_calls) + 1))
    return tool_calls


def coerce_ai_message_tool_markup(message: AIMessage) -> AIMessage:
    """Populate ``message.tool_calls`` from XML-ish markup when providers omit structured calls."""
    if not isinstance(message, AIMessage) or getattr(message, "tool_calls", None):
        return message

    content = message.content
    if not isinstance(content, str):
        return message

    parsed = extract_tool_calls_from_markup(content)
    if parsed:
        message.tool_calls = cast(Any, parsed)
    return message


def _default_start_date(end_date: str) -> str:
    try:
        return (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    except ValueError:
        return end_date


def normalize_tool_args(
    tool_name: str,
    args: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill common ticker/date aliases that local models often omit in XML tool markup."""
    defaults = defaults or {}
    normalized = dict(args or {})

    default_ticker = defaults.get("ticker") or defaults.get("symbol")
    default_date = defaults.get("trade_date") or defaults.get("curr_date") or defaults.get("end_date")
    default_start_date = defaults.get("start_date") or (_default_start_date(default_date) if default_date else None)

    symbol_tools = {
        "get_stock_data",
        "get_intraday_bars",
        "get_session_bars",
        "get_ticker_snapshot",
        "get_last_trade",
        "get_nbbo_quotes",
        "get_options_chain",
        "get_recent_earnings_anchor",
        "get_indicators",
    }
    ticker_tools = {
        "get_news",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_insider_transactions",
    }

    if tool_name in symbol_tools:
        if "symbol" not in normalized:
            normalized["symbol"] = (
                normalized.get("ticker")
                or normalized.get("query")
                or normalized.get("instrument")
                or default_ticker
            )
        normalized.pop("ticker", None)
        normalized.pop("query", None)
        normalized.pop("instrument", None)

    if tool_name in ticker_tools:
        if "ticker" not in normalized:
            normalized["ticker"] = (
                normalized.get("symbol")
                or normalized.get("query")
                or normalized.get("instrument")
                or default_ticker
            )
        normalized.pop("symbol", None)
        normalized.pop("query", None)
        normalized.pop("instrument", None)

    if tool_name == "get_news":
        normalized.setdefault("start_date", defaults.get("start_date") or default_start_date)
        normalized.setdefault("end_date", defaults.get("end_date") or default_date)
    elif tool_name == "get_global_news":
        normalized.setdefault("curr_date", defaults.get("curr_date") or default_date)
        normalized.pop("ticker", None)
        normalized.pop("symbol", None)
        normalized.pop("query", None)
    elif tool_name == "get_stock_data":
        normalized.setdefault("start_date", defaults.get("start_date") or default_start_date)
        normalized.setdefault("end_date", defaults.get("end_date") or default_date)
    elif tool_name in {"get_intraday_bars", "get_session_bars", "get_options_chain"}:
        normalized.setdefault("trade_date", defaults.get("trade_date") or default_date)
    elif tool_name in {
        "get_market_regime",
        "get_recent_earnings_anchor",
        "get_indicators",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    }:
        normalized.setdefault("curr_date", defaults.get("curr_date") or default_date)
        if tool_name == "get_market_regime":
            normalized.pop("ticker", None)
            normalized.pop("symbol", None)
            normalized.pop("query", None)

    return {key: value for key, value in normalized.items() if value is not None}


def invoke_bound_tools_until_completion(
    chain,
    initial_messages,
    *,
    tools,
    max_rounds: int = 6,
    default_tool_args: dict[str, Any] | None = None,
):
    """Run a tool-bound analyst chain until it returns a final prose answer."""
    tool_map = {tool.name: tool for tool in tools}
    messages = list(initial_messages)
    latest: AIMessage | None = None

    for _ in range(max_rounds):
        latest = coerce_ai_message_tool_markup(chain.invoke(messages))
        messages.append(latest)
        tool_calls = getattr(latest, "tool_calls", None) or []
        if not tool_calls:
            return latest

        for idx, tool_call in enumerate(tool_calls, start=1):
            name = tool_call.get("name")
            args = normalize_tool_args(name, tool_call.get("args", {}), default_tool_args) if name else {}
            tool = tool_map.get(name)
            if tool is None:
                result = f"Tool not found: {name}"
            else:
                try:
                    result = tool.invoke(args)
                except Exception as exc:  # pragma: no cover
                    result = f"Tool {name} failed: {exc}"
            messages.append(
                ToolMessage(
                    content=result if isinstance(result, str) else str(result),
                    tool_call_id=tool_call.get("id", f"call_{idx}"),
                    name=name or f"tool_{idx}",
                )
            )

    if latest is None:  # pragma: no cover
        raise RuntimeError("tool-bound chain produced no AI message")
    return latest


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
