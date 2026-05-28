from examples.run_fmp_mcp_stack import build_config


def test_build_config_uses_fmp_mcp_stack():
    config = build_config()

    assert config["llm_provider"] == "openai"
    assert config["backend_url"] == "http://10.17.17.99:8005/v1"
    assert config["deep_think_llm"] == "Qwen/Qwen3.6-27B-FP8"
    assert config["quick_think_llm"] == "Qwen/Qwen3.6-27B-FP8"
    assert config["market_data_mcp_url"] == "https://10.17.17.90:8083/mcp"
    assert config["fmp_mcp_url"] == "http://10.17.17.90:8086/mcp"
    assert config["news_mcp_url"] == "http://10.17.17.90:9081/mcp"
    assert config["mcp_verify_tls"] is False
    assert config["data_vendors"]["core_stock_apis"] == "massive"
    assert config["data_vendors"]["fundamental_data"] == "fmp"
    assert config["data_vendors"]["news_data"] == "fmp"
    assert config["tool_vendors"]["get_stock_data"] == "massive"
    assert config["tool_vendors"]["get_news"] == "fmp,grok"
    assert config["tool_vendors"]["get_global_news"] == "grok,fmp"
    assert config["tool_vendors"]["get_insider_transactions"] == "fmp"
    assert config["data_vendors"]["technical_indicators"] == "massive"


def test_build_config_honors_gemini_llm_env(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "google")
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_LLM", "gemini-3.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "gemini-3.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_LLM_BACKEND_URL", "")

    config = build_config()

    assert config["llm_provider"] == "google"
    assert config["backend_url"] is None
    assert config["deep_think_llm"] == "gemini-3.5-flash"
    assert config["quick_think_llm"] == "gemini-3.5-flash"
