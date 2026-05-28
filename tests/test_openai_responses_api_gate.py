"""Regression tests for OpenAI-compatible local Qwen tool-calling quirks."""

from pydantic import BaseModel
from pydantic import SecretStr

from tradingagents.llm_clients import capabilities as caps_mod
from tradingagents.llm_clients import openai_client as mod


def _capture_constructor_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod,
        "NormalizedChatOpenAI",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


def test_native_openai_uses_responses_api(monkeypatch):
    captured = _capture_constructor_kwargs(monkeypatch)

    mod.OpenAIClient(model="gpt-5.4", provider="openai", api_key="x").get_llm()

    assert captured["kwargs"]["use_responses_api"] is True


def test_custom_openai_compatible_base_url_stays_on_chat_completions(monkeypatch):
    captured = _capture_constructor_kwargs(monkeypatch)

    mod.OpenAIClient(
        model="Qwen/Qwen3.6-27B-FP8",
        provider="openai",
        base_url="http://10.17.17.99:8005/v1",
        api_key="x",
    ).get_llm()

    assert "use_responses_api" not in captured["kwargs"]
    assert captured["kwargs"]["base_url"] == "http://10.17.17.99:8005/v1"


def test_local_qwen_capabilities_disable_tool_choice():
    caps = caps_mod.get_capabilities("Qwen/Qwen3.6-27B-FP8")

    assert caps.supports_tool_choice is False
    assert caps.preferred_structured_method == "function_calling"


def test_bind_tools_falls_back_to_plain_bind_for_local_qwen(monkeypatch):
    captured: dict = {}

    def fake_bind(self, **kwargs):
        captured["kwargs"] = kwargs
        return "bound"

    def fake_bind_tools(self, tools, **kwargs):  # pragma: no cover - should not be used
        raise AssertionError("bind_tools should not be used for local Qwen fallback")

    monkeypatch.setattr(mod.ChatOpenAI, "bind", fake_bind)
    monkeypatch.setattr(mod.ChatOpenAI, "bind_tools", fake_bind_tools)

    llm = mod.NormalizedChatOpenAI(
        model="Qwen/Qwen3.6-27B-FP8",
        api_key=SecretStr("x"),
        base_url="http://10.17.17.99:8005/v1",
    )

    result = llm.bind_tools([object()])

    assert result == "bound"
    assert captured["kwargs"] == {}


def test_structured_output_keeps_schema_tool_for_local_qwen():
    class Pick(BaseModel):
        action: str

    llm = mod.NormalizedChatOpenAI(
        model="Qwen/Qwen3.6-27B-FP8",
        api_key=SecretStr("x"),
        base_url="http://10.17.17.99:8005/v1",
    )

    bound = llm.with_structured_output(Pick)
    first = bound.steps[0] if hasattr(bound, "steps") else bound
    kwargs = getattr(first, "kwargs", {})

    assert "tool_choice" not in kwargs
    assert "strict" not in kwargs
    assert any(
        tool.get("function", {}).get("name") == "Pick"
        for tool in kwargs.get("tools", [])
    )
