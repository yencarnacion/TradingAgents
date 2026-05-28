from __future__ import annotations

import importlib

from langchain_core.messages import AIMessage, ToolMessage

from tradingagents.agents.utils.agent_utils import extract_tool_calls_from_markup


class _FakePrompt:
    def partial(self, **kwargs):
        return self

    def __or__(self, other):
        return other


class _FakeChatPromptTemplate:
    @staticmethod
    def from_messages(_messages):
        return _FakePrompt()


class _SequenceLLM:
    def __init__(self, results: list[AIMessage]):
        self._results = list(results)
        self.invocations = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.invocations.append(list(messages))
        if not self._results:
            raise AssertionError("No fake LLM results remaining")
        return self._results.pop(0)


class _FakeTool:
    def __init__(self, name: str, result: str):
        self.name = name
        self.result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.result


def _state() -> dict:
    return {
        "messages": [("human", "QQQ")],
        "company_of_interest": "QQQ",
        "trade_date": "2026-05-26",
        "asset_type": "stock",
    }


def test_extracts_batched_json_calls_from_fenced_block():
    calls = extract_tool_calls_from_markup(
        """I'll gather comprehensive fundamental information.

```json
{"calls": [
  {"call": {"tool": "get_fundamentals"}},
  {"call": {"tool": "get_balance_sheet"}},
  {"call": {"tool": "get_cashflow"}},
  {"call": {"tool": "get_income_statement"}}
]}
```"""
    )

    assert [call["name"] for call in calls] == [
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    ]
    assert all(call["args"] == {} for call in calls)


def test_extracts_simple_calls_from_repeated_unclosed_tool_call_markers():
    calls = extract_tool_calls_from_markup(
        """<tool_call>
<tool_call>
get_global_news
<tool_call>
<tool_call>
get_news
<tool_call>
<tool_call>
get_global_news(curr_date="2026-05-27", look_back_days=7, limit=10)
<tool_call>
get_news(query="SPY S&P 500 market news", start_date="2026-05-20", end_date="2026-05-27")"""
    )

    assert calls == [
        {
            "name": "get_global_news",
            "args": {"curr_date": "2026-05-27", "look_back_days": 7, "limit": 10},
            "id": "call_1",
            "type": "tool_call",
        },
        {
            "name": "get_news",
            "args": {
                "query": "SPY S&P 500 market news",
                "start_date": "2026-05-20",
                "end_date": "2026-05-27",
            },
            "id": "call_2",
            "type": "tool_call",
        },
    ]


def test_news_analyst_executes_markup_tool_call_and_returns_final_report(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.news_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    news_tool = _FakeTool("get_news", "headline payload")
    global_news_tool = _FakeTool("get_global_news", "macro payload")
    monkeypatch.setattr(module, "get_news", news_tool)
    monkeypatch.setattr(module, "get_global_news", global_news_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "<tool_call>\n"
                    "<function=get_news>\n"
                    "<parameter=ticker>\nQQQ\n</parameter>\n"
                    "<parameter=start_date>\n2026-05-19\n</parameter>\n"
                    "<parameter=end_date>\n2026-05-26\n</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            ),
            AIMessage(content="## News Report\n\nMacro tone improved after catalyst review."),
        ]
    )

    node = module.create_news_analyst(llm)
    output = node(_state())

    assert output["news_report"] == "## News Report\n\nMacro tone improved after catalyst review."
    assert news_tool.calls == [{"ticker": "QQQ", "start_date": "2026-05-19", "end_date": "2026-05-26"}]
    assert global_news_tool.calls == []
    assert isinstance(llm.invocations[1][-1], ToolMessage)


def test_news_analyst_executes_qwen_invoke_markup(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.news_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    news_tool = _FakeTool("get_news", "headline payload")
    global_news_tool = _FakeTool("get_global_news", "macro payload")
    monkeypatch.setattr(module, "get_news", news_tool)
    monkeypatch.setattr(module, "get_global_news", global_news_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "<tool_code>\n"
                    "<function_calls>\n"
                    "<invoke name='get_global_news'>\n"
                    "<parameter name='curr_date'>2026-05-26</parameter>\n"
                    "<parameter name='look_back_days'>7</parameter>\n"
                    "</invoke>\n"
                    "<invoke name='get_news'>\n"
                    "<parameter name='query'>QQQ</parameter>\n"
                    "<parameter name='start_date'>2026-05-19</parameter>\n"
                    "<parameter name='end_date'>2026-05-26</parameter>\n"
                    "</invoke>\n"
                    "</function_calls>\n"
                    "</tool_code>"
                )
            ),
            AIMessage(content="## News Report\n\nMacro tone improved after catalyst review."),
        ]
    )

    node = module.create_news_analyst(llm)
    output = node(_state())

    assert output["news_report"] == "## News Report\n\nMacro tone improved after catalyst review."
    assert global_news_tool.calls == [{"curr_date": "2026-05-26", "look_back_days": 7}]
    assert news_tool.calls == [{"ticker": "QQQ", "start_date": "2026-05-19", "end_date": "2026-05-26"}]
    assert isinstance(llm.invocations[1][-2], ToolMessage)
    assert isinstance(llm.invocations[1][-1], ToolMessage)


def test_market_analyst_keeps_final_written_report(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.market_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    final_report = AIMessage(content="## Market Report\n\nTape is constructive.")
    node = module.create_market_analyst(_SequenceLLM([final_report]))
    output = node(_state())

    assert output["market_report"] == "## Market Report\n\nTape is constructive."
    assert output["messages"][0].tool_calls == []


def test_market_analyst_executes_recent_earnings_anchor_tool(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.market_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    earnings_tool = _FakeTool("get_recent_earnings_anchor", "earnings anchor payload")
    monkeypatch.setattr(module, "get_recent_earnings_anchor", earnings_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "<tool_call>\n"
                    "<function=get_recent_earnings_anchor>\n"
                    "<parameter=symbol>\nQQQ\n</parameter>\n"
                    "<parameter=curr_date>\n2026-05-26\n</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            ),
            AIMessage(content="## Market Report\n\nPrice is holding above post-earnings AVWAP."),
        ]
    )

    node = module.create_market_analyst(llm)
    output = node(_state())

    assert output["market_report"] == "## Market Report\n\nPrice is holding above post-earnings AVWAP."
    assert earnings_tool.calls == [{"symbol": "QQQ", "curr_date": "2026-05-26"}]
    assert isinstance(llm.invocations[1][-1], ToolMessage)


def test_market_analyst_executes_malformed_tool_call_without_closing_wrapper(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.market_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    options_tool = _FakeTool("get_options_chain", "options payload")
    monkeypatch.setattr(module, "get_options_chain", options_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "<tool_call>\n"
                    "<function=get_options_chain>\n"
                    "</function>"
                )
            ),
            AIMessage(content="## Market Report\n\nOptions context was reviewed."),
        ]
    )

    node = module.create_market_analyst(llm)
    output = node(_state())

    assert output["market_report"] == "## Market Report\n\nOptions context was reviewed."
    assert options_tool.calls == [{"symbol": "QQQ", "trade_date": "2026-05-26"}]
    assert isinstance(llm.invocations[1][-1], ToolMessage)


def test_market_analyst_executes_qwen_invoke_markup_with_defaults(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.market_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    market_regime_tool = _FakeTool("get_market_regime", "regime payload")
    snapshot_tool = _FakeTool("get_ticker_snapshot", "snapshot payload")
    intraday_tool = _FakeTool("get_intraday_bars", "intraday payload")
    session_tool = _FakeTool("get_session_bars", "session payload")
    stock_data_tool = _FakeTool("get_stock_data", "stock data payload")
    options_tool = _FakeTool("get_options_chain", "options payload")

    monkeypatch.setattr(module, "get_market_regime", market_regime_tool)
    monkeypatch.setattr(module, "get_ticker_snapshot", snapshot_tool)
    monkeypatch.setattr(module, "get_intraday_bars", intraday_tool)
    monkeypatch.setattr(module, "get_session_bars", session_tool)
    monkeypatch.setattr(module, "get_stock_data", stock_data_tool)
    monkeypatch.setattr(module, "get_options_chain", options_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "<tool_code>\n"
                    "<function_calls>\n"
                    "<invoke name='get_market_regime'><parameter name='ticker'>QQQ</parameter></invoke>\n"
                    "<invoke name='get_ticker_snapshot'><parameter name='ticker'>QQQ</parameter></invoke>\n"
                    "<invoke name='get_intraday_bars'><parameter name='ticker'>QQQ</parameter></invoke>\n"
                    "<invoke name='get_session_bars'><parameter name='ticker'>QQQ</parameter></invoke>\n"
                    "<invoke name='get_stock_data'><parameter name='ticker'>QQQ</parameter></invoke>\n"
                    "<invoke name='get_options_chain'><parameter name='ticker'>QQQ</parameter></invoke>\n"
                    "</function_calls>\n"
                    "</tool_code>"
                )
            ),
            AIMessage(content="## Market Report\n\nSpot is near 750 and tape is constructive."),
        ]
    )

    node = module.create_market_analyst(llm)
    output = node(_state())

    assert output["market_report"] == "## Market Report\n\nSpot is near 750 and tape is constructive."
    assert market_regime_tool.calls == [{"curr_date": "2026-05-26"}]
    assert snapshot_tool.calls == [{"symbol": "QQQ"}]
    assert intraday_tool.calls == [{"symbol": "QQQ", "trade_date": "2026-05-26"}]
    assert session_tool.calls == [{"symbol": "QQQ", "trade_date": "2026-05-26"}]
    assert stock_data_tool.calls == [{"symbol": "QQQ", "start_date": "2025-05-26", "end_date": "2026-05-26"}]
    assert options_tool.calls == [{"symbol": "QQQ", "trade_date": "2026-05-26"}]
    assert all(isinstance(message, ToolMessage) for message in llm.invocations[1][-6:])


def test_market_analyst_executes_fenced_json_tool_plan(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.market_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    market_regime_tool = _FakeTool("get_market_regime", "regime payload")
    snapshot_tool = _FakeTool("get_ticker_snapshot", "snapshot payload")
    intraday_tool = _FakeTool("get_intraday_bars", "intraday payload")
    session_tool = _FakeTool("get_session_bars", "session payload")
    options_tool = _FakeTool("get_options_chain", "options payload")
    last_trade_tool = _FakeTool("get_last_trade", "last trade payload")
    quote_tool = _FakeTool("get_nbbo_quotes", "quote payload")
    stock_data_tool = _FakeTool("get_stock_data", "stock data payload")

    monkeypatch.setattr(module, "get_market_regime", market_regime_tool)
    monkeypatch.setattr(module, "get_ticker_snapshot", snapshot_tool)
    monkeypatch.setattr(module, "get_intraday_bars", intraday_tool)
    monkeypatch.setattr(module, "get_session_bars", session_tool)
    monkeypatch.setattr(module, "get_options_chain", options_tool)
    monkeypatch.setattr(module, "get_last_trade", last_trade_tool)
    monkeypatch.setattr(module, "get_nbbo_quotes", quote_tool)
    monkeypatch.setattr(module, "get_stock_data", stock_data_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "I will pull the market data.\n\n"
                    "```json\n"
                    "[\n"
                    '  {"tool": "get_market_regime", "args": {"ticker": "QQQ"}},\n'
                    '  {"tool": "get_ticker_snapshot", "args": {"ticker": "QQQ"}},\n'
                    '  {"tool": "get_intraday_bars", "args": {"ticker": "QQQ"}},\n'
                    '  {"tool": "get_session_bars", "args": {"ticker": "QQQ"}},\n'
                    '  {"tool": "get_options_chain", "args": {"ticker": "QQQ"}},\n'
                    '  {"tool": "get_last_trade", "args": {"ticker": "QQQ"}},\n'
                    '  {"tool": "get_nbbo_quotes", "args": {"ticker": "QQQ"}},\n'
                    '  {"tool": "get_stock_data", "args": {"ticker": "QQQ"}}\n'
                    "]\n"
                    "```"
                )
            ),
            AIMessage(content="## Market Report\n\nTape is constructive after tool review."),
        ]
    )

    node = module.create_market_analyst(llm)
    output = node(_state())

    assert output["market_report"] == "## Market Report\n\nTape is constructive after tool review."
    assert market_regime_tool.calls == [{"curr_date": "2026-05-26"}]
    assert snapshot_tool.calls == [{"symbol": "QQQ"}]
    assert intraday_tool.calls == [{"symbol": "QQQ", "trade_date": "2026-05-26"}]
    assert session_tool.calls == [{"symbol": "QQQ", "trade_date": "2026-05-26"}]
    assert options_tool.calls == [{"symbol": "QQQ", "trade_date": "2026-05-26"}]
    assert last_trade_tool.calls == [{"symbol": "QQQ"}]
    assert quote_tool.calls == [{"symbol": "QQQ"}]
    assert stock_data_tool.calls == [{"symbol": "QQQ", "start_date": "2025-05-26", "end_date": "2026-05-26"}]
    assert all(isinstance(message, ToolMessage) for message in llm.invocations[1][-8:])


def test_news_analyst_executes_fenced_simple_call(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.news_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    news_tool = _FakeTool("get_news", "headline payload")
    global_news_tool = _FakeTool("get_global_news", "macro payload")
    monkeypatch.setattr(module, "get_news", news_tool)
    monkeypatch.setattr(module, "get_global_news", global_news_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "Let me pull targeted news.\n\n"
                    '```json\ncall_get_news(query="QQQ", start_date="2026-05-19", end_date="2026-05-26")\n```'
                )
            ),
            AIMessage(content="## News Report\n\nMacro tone improved after catalyst review."),
        ]
    )

    node = module.create_news_analyst(llm)
    output = node(_state())

    assert output["news_report"] == "## News Report\n\nMacro tone improved after catalyst review."
    assert news_tool.calls == [{"ticker": "QQQ", "start_date": "2026-05-19", "end_date": "2026-05-26"}]
    assert global_news_tool.calls == []


def test_fundamentals_analyst_executes_fenced_action_json(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.fundamentals_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    fundamentals_tool = _FakeTool("get_fundamentals", "fundamentals payload")
    monkeypatch.setattr(module, "get_fundamentals", fundamentals_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "Let me gather comprehensive fundamental data.\n\n"
                    "```json\n"
                    "{\n"
                    '  "action": {\n'
                    '    "function": "get_fundamentals",\n'
                    '    "parameters": {"instrument": "QQQ"}\n'
                    "  }\n"
                    "}\n"
                    "```"
                )
            ),
            AIMessage(content="## Fundamentals Report\n\nETF fundamentals reviewed."),
        ]
    )

    node = module.create_fundamentals_analyst(llm)
    output = node(_state())

    assert output["fundamentals_report"] == "## Fundamentals Report\n\nETF fundamentals reviewed."
    assert fundamentals_tool.calls == [{"ticker": "QQQ", "curr_date": "2026-05-26"}]


def test_fundamentals_analyst_executes_tool_with_numeric_arguments(monkeypatch):
    module = importlib.import_module("tradingagents.agents.analysts.fundamentals_analyst")
    monkeypatch.setattr(module, "ChatPromptTemplate", _FakeChatPromptTemplate)

    fundamentals_tool = _FakeTool("get_fundamentals", "fundamentals payload")
    balance_sheet_tool = _FakeTool("get_balance_sheet", "balance payload")
    cashflow_tool = _FakeTool("get_cashflow", "cashflow payload")
    income_tool = _FakeTool("get_income_statement", "income payload")
    monkeypatch.setattr(module, "get_fundamentals", fundamentals_tool)
    monkeypatch.setattr(module, "get_balance_sheet", balance_sheet_tool)
    monkeypatch.setattr(module, "get_cashflow", cashflow_tool)
    monkeypatch.setattr(module, "get_income_statement", income_tool)

    llm = _SequenceLLM(
        [
            AIMessage(
                content=(
                    "<tool_call>\n"
                    "<function=get_balance_sheet>\n"
                    "<parameter=ticker>\nQQQ\n</parameter>\n"
                    "<parameter=curr_date>\n2026-05-26\n</parameter>\n"
                    "<parameter=freq>\nquarterly\n</parameter>\n"
                    "<parameter=limit>\n50\n</parameter>\n"
                    "</function>\n"
                    "</tool_call>"
                )
            ),
            AIMessage(content="## Fundamentals Report\n\nBalance sheet remained resilient."),
        ]
    )

    node = module.create_fundamentals_analyst(llm)
    output = node(_state())

    assert output["fundamentals_report"] == "## Fundamentals Report\n\nBalance sheet remained resilient."
    assert balance_sheet_tool.calls == [
        {
            "ticker": "QQQ",
            "curr_date": "2026-05-26",
            "freq": "quarterly",
            "limit": 50,
        }
    ]
    assert fundamentals_tool.calls == []
    assert cashflow_tool.calls == []
    assert income_tool.calls == []
