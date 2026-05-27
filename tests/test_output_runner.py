import argparse
import json
from datetime import date, datetime
from unittest.mock import patch

from examples.run_grounded_stack import resolve_inputs
from tradingagents.output_runner import (
    FINAL_BEGIN,
    FINAL_END,
    RunPaths,
    build_app_command,
    build_run_slug,
    extract_final_decision,
    format_duration,
    parse_args,
    publish_runtime_artifacts,
    render_index_html,
    render_live_html,
    render_markdown_html,
    resolve_run_request,
    state_log_path,
    write_report_artifacts,
    write_technical_chart_artifact,
    write_technical_indicators_artifact,
)


def test_resolve_inputs_defaults_to_spy_and_today(monkeypatch):
    monkeypatch.delenv("TICKER", raising=False)
    monkeypatch.delenv("ANALYSIS_DATE", raising=False)

    ticker, analysis_date = resolve_inputs()

    assert ticker == "SPY"
    assert analysis_date == date.today().isoformat()


def test_resolve_run_request_uses_provided_values():
    ticker, analysis_date = resolve_run_request("nvda", "2026-05-25", today=date(2026, 1, 1))

    assert ticker == "NVDA"
    assert analysis_date == "2026-05-25"


def test_format_duration_hms():
    assert format_duration(3661) == "01:01:01"


def test_extract_final_decision_from_markers():
    text = f"hello\n{FINAL_BEGIN}\n# Title\n\nBody\n{FINAL_END}\nbye\n"
    assert extract_final_decision(text) == "# Title\n\nBody"


def test_build_run_slug_contains_ticker_and_date():
    slug = build_run_slug("SPY", "2026-05-25", datetime(2026, 5, 25, 12, 34, 56))
    assert "spy" in slug
    assert "2026-05-25" in slug


def test_build_app_command_switches_examples(tmp_path):
    grounded = build_app_command(tmp_path, "SPY", "2026-05-25", "grounded")
    fmp = build_app_command(tmp_path, "SPY", "2026-05-25", "fmp")

    assert grounded[-3:] == [str(tmp_path / "examples" / "run_grounded_stack.py"), "SPY", "2026-05-25"]
    assert fmp[-3:] == [str(tmp_path / "examples" / "run_fmp_mcp_stack.py"), "SPY", "2026-05-25"]


def test_parse_args_defaults_to_fmp_stack_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("TICKER_AGENTS_STACK", raising=False)
    args = parse_args([])

    assert args.stack == "fmp"


def test_parse_args_uses_env_default_stack(monkeypatch):
    monkeypatch.setenv("TICKER_AGENTS_STACK", "grounded")
    args = parse_args([])

    assert args.stack == "grounded"


def test_render_live_html_includes_structured_view_hooks():
    html = render_live_html("Ticker Agents Run: SPY")

    assert "Live structured view" in html
    assert "entry-human" in html
    assert "entry-tool" in html
    assert "entry-final" in html
    assert "parseStructuredEntries" in html
    assert "Tool calls" in html
    assert "Published reports" in html
    assert "renderReportArtifacts" in html
    assert "reportGroups" in html


def test_render_markdown_html_falls_back_when_markdown_missing():
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "markdown":
            raise ModuleNotFoundError("No module named 'markdown'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        html = render_markdown_html("Final Decision", "# Title\n\n- item")

    assert "plain-text fallback view" in html
    assert "&lt;" not in html  # sanity check: no stray HTML injection from markdown body
    assert "# Title" in html


def test_write_report_artifacts_publishes_bull_bear_and_risk_reports(tmp_path):
    state = {
        "sentiment_report": "# Sentiment\n\nBullish crowding.",
        "investment_debate_state": {
            "bull_history": "## Bull Case\n\nMomentum + flows.",
            "bear_history": "## Bear Case\n\nCrowding + macro risk.",
            "judge_decision": "## Research Manager\n\nScale in.",
        },
        "investment_plan": "## Investment Plan\n\nAdd on pullbacks.",
        "trader_investment_decision": "## Trader\n\nBuy in tranches.",
        "risk_debate_state": {
            "aggressive_history": "## Aggressive\n\nLean in.",
            "conservative_history": "## Conservative\n\nProtect capital.",
            "neutral_history": "## Neutral\n\nStagger entries.",
            "judge_decision": "## Portfolio Manager\n\nOverweight with guardrails.",
        },
    }

    artifacts = write_report_artifacts(
        tmp_path,
        "QQQ",
        "2026-05-25",
        state,
        public_base_url="http://10.17.17.98:8765/test-run",
    )
    slugs = {artifact["slug"] for artifact in artifacts}

    assert "sentiment-report" in slugs
    assert "bullish-report" in slugs
    assert "bearish-report" in slugs
    assert "research-manager-report" in slugs
    assert "trader-report" in slugs
    assert "portfolio-manager-report" in slugs
    assert (tmp_path / "bullish-report.md").exists()
    assert (tmp_path / "bullish-report.html").exists()
    assert "Bull Case" in (tmp_path / "bullish-report.md").read_text()
    bullish = next(artifact for artifact in artifacts if artifact["slug"] == "bullish-report")
    assert bullish["category"] == "debate"
    assert bullish["html_url"] == "http://10.17.17.98:8765/test-run/bullish-report.html"
    assert bullish["markdown_url"] == "http://10.17.17.98:8765/test-run/bullish-report.md"


def test_write_technical_indicators_artifact_publishes_separate_report(tmp_path, monkeypatch):
    from tradingagents import output_runner

    monkeypatch.setattr(output_runner, "_build_stack_config", lambda stack: {"data_vendors": {"technical_indicators": "yfinance"}, "tool_vendors": {}})
    monkeypatch.setattr(output_runner, "set_config", lambda config: None)
    monkeypatch.setattr(output_runner, "get_vendor", lambda category, method=None: "yfinance")
    monkeypatch.setattr(
        output_runner,
        "route_to_vendor",
        lambda method, ticker, indicator, analysis_date, look_back_days: (
            f"## {indicator} values from 2026-04-25 to {analysis_date}:\n\n"
            f"{analysis_date}: {indicator}-value"
        ),
    )

    artifact = write_technical_indicators_artifact(
        tmp_path,
        "QQQ",
        "2026-05-25",
        stack="fmp",
        public_base_url="http://10.17.17.98:8765/test-run",
    )

    assert artifact["slug"] == "technical-indicators-report"
    assert artifact["category"] == "technical"
    body = (tmp_path / "technical-indicators-report.md").read_text()
    assert "# Technical Indicators Report" in body
    assert "Configured vendor" in body
    assert "## RSI (`rsi`)" in body
    assert "rsi-value" in body
    assert (tmp_path / "technical-indicators-report.html").exists()


def test_write_technical_chart_artifact_uses_massive_data_and_lightweight_charts(tmp_path, monkeypatch):
    from tradingagents import output_runner

    monkeypatch.setattr(output_runner, "_build_stack_config", lambda stack: {"data_vendors": {}, "tool_vendors": {}})
    monkeypatch.setattr(output_runner, "set_config", lambda config: None)
    monkeypatch.setattr(
        output_runner,
        "load_recent_earnings_anchor",
        lambda ticker, analysis_date, stack: {
            "anchor_date": "2026-05-21",
            "anchor_label": "Most recent earnings (AMC 2026-05-21)",
            "source_vendor": "fmp",
        },
    )
    monkeypatch.setattr(
        output_runner,
        "route_to_vendor",
        lambda method, ticker, start_date, analysis_date: (
            "# Stock data for QQQ\n"
            "date,open,high,low,close,volume,vwap\n"
            "2026-05-20,500,505,495,503,1000000,501\n"
            "2026-05-21,503,507,500,506,1200000,504\n"
            "2026-05-22,506,510,504,509,1300000,507\n"
            "2026-05-23,509,512,507,511,1100000,510\n"
            "2026-05-26,511,515,509,514,1250000,512\n"
        ),
    )

    artifact = write_technical_chart_artifact(
        tmp_path,
        "QQQ",
        "2026-05-26",
        stack="fmp",
        public_base_url="http://10.17.17.98:8765/test-run",
    )

    assert artifact["slug"] == "technical-chart"
    assert artifact["category"] == "technical"
    body = (tmp_path / "technical-chart.md").read_text()
    assert "# Technical Chart" in body
    html = (tmp_path / "technical-chart.html").read_text()
    assert "lightweight-charts" in html
    assert "Massive MCP" in html
    assert "priceChart" in html
    assert "priceLegend" in html
    assert "rsiLegend" in html
    assert "macdLegend" in html
    assert "legend-item" in html
    assert "lastValueVisible: false" in html
    assert "priceLineVisible: false" in html
    assert "title: 'MACD'" not in html
    assert "title: 'Signal'" not in html
    assert "title, lastValueVisible: false" not in html
    assert "rsiChart" in html
    assert "macdChart" in html
    start = html.index('<script id="chart-data" type="application/json">') + len('<script id="chart-data" type="application/json">')
    end = html.index('</script>', start)
    payload = html[start:end]
    assert '&quot;' not in payload
    parsed = json.loads(payload)
    assert parsed["source_vendor"] == "massive"
    assert parsed["candles"][-1]["close"] == 514.0
    assert parsed["earnings_anchor"]["anchor_date"] == "2026-05-21"
    assert parsed["avwap_from_earnings"][-1]["value"] is not None
    assert "AVWAP from earnings" in body
    assert "Most recent earnings (AMC 2026-05-21)" in html


def test_write_technical_chart_artifact_skips_earnings_avwap_for_etf_like_tickers(tmp_path, monkeypatch):
    from tradingagents import output_runner

    monkeypatch.setattr(output_runner, "_build_stack_config", lambda stack: {"data_vendors": {}, "tool_vendors": {}})
    monkeypatch.setattr(output_runner, "set_config", lambda config: None)
    monkeypatch.setattr(output_runner, "load_recent_earnings_anchor", lambda ticker, analysis_date, stack: {})
    monkeypatch.setattr(
        output_runner,
        "route_to_vendor",
        lambda method, ticker, start_date, analysis_date: (
            "# Stock data for QQQ\n"
            "date,open,high,low,close,volume,vwap\n"
            "2026-05-20,500,505,495,503,1000000,501\n"
            "2026-05-21,503,507,500,506,1200000,504\n"
            "2026-05-22,506,510,504,509,1300000,507\n"
            "2026-05-23,509,512,507,511,1100000,510\n"
            "2026-05-26,511,515,509,514,1250000,512\n"
        ),
    )

    write_technical_chart_artifact(
        tmp_path,
        "QQQ",
        "2026-05-26",
        stack="fmp",
        public_base_url="http://10.17.17.98:8765/test-run",
    )

    body = (tmp_path / "technical-chart.md").read_text()
    html = (tmp_path / "technical-chart.html").read_text()
    start = html.index('<script id="chart-data" type="application/json">') + len('<script id="chart-data" type="application/json">')
    end = html.index('</script>', start)
    parsed = json.loads(html[start:end])

    assert parsed["earnings_anchor"] == {}
    assert parsed["avwap_from_earnings"] == []
    assert "AVWAP from earnings" not in body


def test_render_index_html_lists_report_artifacts():
    html = render_index_html(
        {
            "title": "Ticker Agents Run: QQQ @ 2026-05-25",
            "ticker": "QQQ",
            "analysis_date": "2026-05-25",
            "status": "completed",
            "started_at": "2026-05-25T08:00:00",
            "finished_at": "2026-05-25T08:10:00",
            "duration_hms": "00:10:00",
            "exit_code": 0,
            "command": "python run_fmp_mcp_stack.py QQQ 2026-05-25",
            "has_final_markdown": True,
            "state_log": "full_states_log_2026-05-25.json",
            "report_artifacts": [
                {
                    "title": "Bullish Report",
                    "category": "debate",
                    "markdown_path": "bullish-report.md",
                    "html_path": "bullish-report.html",
                    "markdown_url": "http://10.17.17.98:8765/test-run/bullish-report.md",
                    "html_url": "http://10.17.17.98:8765/test-run/bullish-report.html",
                },
                {
                    "title": "Market Report",
                    "category": "analyst",
                    "markdown_path": "market-report.md",
                    "html_path": "market-report.html",
                    "markdown_url": "http://10.17.17.98:8765/test-run/market-report.md",
                    "html_url": "http://10.17.17.98:8765/test-run/market-report.html",
                },
            ],
        }
    )

    assert "Published reports" in html
    assert "Bullish Report" in html
    assert "bullish-report.html" in html
    assert "full_states_log_2026-05-25.json" in html
    assert "Analyst reports" in html
    assert "Debate reports" in html
    assert "2 reports published" in html
    assert "http://10.17.17.98:8765/test-run/bullish-report.html" in html


def test_state_log_path_uses_tradingagents_home_convention():
    path = state_log_path("QQQ", "2026-05-25")
    assert str(path).endswith("/.tradingagents/logs/QQQ/TradingAgentsStrategy_logs/full_states_log_2026-05-25.json")


def test_render_index_html_formats_started_at_in_new_york_time():
    html = render_index_html(
        {
            "title": "Ticker Agents Run: QQQ @ 2026-05-25",
            "ticker": "QQQ",
            "analysis_date": "2026-05-25",
            "status": "running",
            "started_at": "2026-05-25T08:00:00+00:00",
            "finished_at": None,
            "duration_hms": "00:00:00",
            "exit_code": None,
            "command": "python run_fmp_mcp_stack.py QQQ 2026-05-25",
            "has_final_markdown": False,
            "state_log": None,
            "report_artifacts": [],
        }
    )

    assert "2026-05-25 04:00:00 EDT" in html
    assert "2026-05-25T08:00:00+00:00" not in html


def test_publish_runtime_artifacts_updates_metadata_and_index_mid_run(tmp_path, monkeypatch):
    from tradingagents import output_runner

    run_dir = tmp_path / "qqq-run"
    run_dir.mkdir()
    metadata_path = run_dir / "metadata.json"
    index_path = run_dir / "index.html"
    state_log = tmp_path / "full_states_log_2026-05-25.json"
    metadata = {
        "title": "Ticker Agents Run: QQQ @ 2026-05-25",
        "ticker": "QQQ",
        "analysis_date": "2026-05-25",
        "status": "running",
        "started_at": "2026-05-25T08:00:00+00:00",
        "finished_at": None,
        "duration_hms": "00:00:00",
        "exit_code": None,
        "command": "python run_fmp_mcp_stack.py QQQ 2026-05-25",
        "has_final_markdown": False,
        "state_log": None,
        "report_artifacts": [],
        "artifact_warning": None,
        "stack": "fmp",
    }

    state_log.write_text(
        json.dumps(
            {
                "sentiment_report": "# Sentiment\n\nMixed.",
                "investment_debate_state": {"bull_history": "## Bull\n\nUptrend."},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        output_runner,
        "write_technical_indicators_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-indicators-report",
            "title": "Technical Indicators Report",
            "category": "technical",
            "markdown_path": "technical-indicators-report.md",
            "html_path": "technical-indicators-report.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-run/technical-indicators-report.md",
            "html_url": "http://10.17.17.98:8765/qqq-run/technical-indicators-report.html",
        },
    )
    monkeypatch.setattr(
        output_runner,
        "write_technical_chart_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-chart",
            "title": "Technical Chart",
            "category": "technical",
            "markdown_path": "technical-chart.md",
            "html_path": "technical-chart.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-run/technical-chart.md",
            "html_url": "http://10.17.17.98:8765/qqq-run/technical-chart.html",
        },
    )

    publish_runtime_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        metadata_path=metadata_path,
        index_path=index_path,
        ticker="QQQ",
        analysis_date="2026-05-25",
        state_log=state_log,
        public_base_url="http://10.17.17.98:8765/qqq-run",
    )

    persisted = json.loads(metadata_path.read_text())
    assert persisted["state_log"] == "full_states_log_2026-05-25.json"
    assert any(artifact["slug"] == "bullish-report" for artifact in persisted["report_artifacts"])
    assert any(artifact["slug"] == "technical-indicators-report" for artifact in persisted["report_artifacts"])
    assert any(artifact["slug"] == "technical-chart" for artifact in persisted["report_artifacts"])
    assert persisted["report_artifacts"][0]["html_url"].startswith("http://10.17.17.98:8765/qqq-run/")
    assert (run_dir / "bullish-report.html").exists()
    assert "Published reports" in index_path.read_text()
    assert "Technical reports" in index_path.read_text()


def test_publish_runtime_artifacts_still_publishes_technical_chart_without_state_log(tmp_path, monkeypatch):
    from tradingagents import output_runner

    run_dir = tmp_path / "qqq-run"
    run_dir.mkdir()
    metadata_path = run_dir / "metadata.json"
    index_path = run_dir / "index.html"
    state_log = tmp_path / "missing-state-log.json"
    metadata = {
        "title": "Ticker Agents Run: QQQ @ 2026-05-25",
        "ticker": "QQQ",
        "analysis_date": "2026-05-25",
        "status": "running",
        "started_at": "2026-05-25T08:00:00+00:00",
        "finished_at": None,
        "duration_hms": "00:00:00",
        "exit_code": None,
        "command": "python run_fmp_mcp_stack.py QQQ 2026-05-25",
        "has_final_markdown": False,
        "state_log": None,
        "report_artifacts": [],
        "artifact_warning": None,
        "stack": "fmp",
    }

    monkeypatch.setattr(
        output_runner,
        "write_technical_indicators_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-indicators-report",
            "title": "Technical Indicators Report",
            "category": "technical",
            "markdown_path": "technical-indicators-report.md",
            "html_path": "technical-indicators-report.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-run/technical-indicators-report.md",
            "html_url": "http://10.17.17.98:8765/qqq-run/technical-indicators-report.html",
        },
    )
    monkeypatch.setattr(
        output_runner,
        "write_technical_chart_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-chart",
            "title": "Technical Chart",
            "category": "technical",
            "markdown_path": "technical-chart.md",
            "html_path": "technical-chart.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-run/technical-chart.md",
            "html_url": "http://10.17.17.98:8765/qqq-run/technical-chart.html",
        },
    )

    publish_runtime_artifacts(
        run_dir=run_dir,
        metadata=metadata,
        metadata_path=metadata_path,
        index_path=index_path,
        ticker="QQQ",
        analysis_date="2026-05-25",
        state_log=state_log,
        public_base_url="http://10.17.17.98:8765/qqq-run",
    )

    persisted = json.loads(metadata_path.read_text())
    assert persisted["state_log"] is None
    assert [artifact["slug"] for artifact in persisted["report_artifacts"]] == [
        "technical-indicators-report",
        "technical-chart",
    ]
    assert persisted["report_summary"]["count"] == 2
    assert "Technical Chart" in index_path.read_text()


def test_run_finalizes_metadata_even_if_final_rendering_fails(tmp_path, monkeypatch):
    from tradingagents import output_runner

    run_dir = tmp_path / "qqq-run"
    paths = RunPaths(
        run_dir=run_dir,
        console_txt=run_dir / "console.txt",
        metadata_json=run_dir / "metadata.json",
        live_html=run_dir / "live.html",
        index_html=run_dir / "index.html",
        final_md=run_dir / "final.md",
        final_html=run_dir / "final.html",
    )

    monkeypatch.setattr(
        output_runner,
        "parse_args",
        lambda argv=None: argparse.Namespace(ticker="QQQ", analysis_date="2026-05-25", stack="fmp", port=8765),
    )
    monkeypatch.setattr(output_runner, "resolve_run_request", lambda ticker, analysis_date: ("QQQ", "2026-05-25"))
    monkeypatch.setattr(output_runner, "build_run_slug", lambda *args, **kwargs: "qqq-test-run")
    monkeypatch.setattr(output_runner, "build_paths", lambda run_path: paths)
    monkeypatch.setattr(output_runner, "ensure_http_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(output_runner, "state_log_path", lambda ticker, analysis_date: tmp_path / "missing-state-log.json")

    def fake_stream(command, cwd, env, out_handle, capture, on_output=None):
        if "run_fmp_mcp_stack.py" in " ".join(command):
            payload = f"{FINAL_BEGIN}\nOverweight\n{FINAL_END}\n"
            out_handle.write(payload)
            capture.append(payload)
            if on_output:
                on_output()
        return 0

    monkeypatch.setattr(output_runner, "stream_command", fake_stream)
    monkeypatch.setattr(
        output_runner,
        "write_technical_indicators_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-indicators-report",
            "title": "Technical Indicators Report",
            "category": "technical",
            "markdown_path": "technical-indicators-report.md",
            "html_path": "technical-indicators-report.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-test-run/technical-indicators-report.md",
            "html_url": "http://10.17.17.98:8765/qqq-test-run/technical-indicators-report.html",
        },
    )
    monkeypatch.setattr(
        output_runner,
        "write_technical_chart_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-chart",
            "title": "Technical Chart",
            "category": "technical",
            "markdown_path": "technical-chart.md",
            "html_path": "technical-chart.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-test-run/technical-chart.md",
            "html_url": "http://10.17.17.98:8765/qqq-test-run/technical-chart.html",
        },
    )
    monkeypatch.setattr(output_runner, "render_markdown_html", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    exit_code = output_runner.run([])

    assert exit_code == 0
    metadata = json.loads(paths.metadata_json.read_text())
    assert metadata["status"] == "completed_with_warnings"
    assert metadata["stack"] == "fmp"
    assert metadata["command"].endswith("run_fmp_mcp_stack.py QQQ 2026-05-25")
    assert metadata["exit_code"] == 0
    assert metadata["finished_at"] is not None
    assert metadata["has_final_markdown"] is True
    assert "final artifact rendering failed: boom" == metadata["artifact_warning"]
    assert paths.final_md.read_text().strip() == "Overweight"
    summary = paths.console_txt.read_text()
    assert "# Run summary" in summary
    assert "status: completed_with_warnings" in summary
    assert "artifact_warning: final artifact rendering failed: boom" in summary
    assert "report_artifacts: 2" in summary


def test_run_publishes_state_log_reports(tmp_path, monkeypatch):
    from tradingagents import output_runner

    run_dir = tmp_path / "qqq-run"
    paths = RunPaths(
        run_dir=run_dir,
        console_txt=run_dir / "console.txt",
        metadata_json=run_dir / "metadata.json",
        live_html=run_dir / "live.html",
        index_html=run_dir / "index.html",
        final_md=run_dir / "final.md",
        final_html=run_dir / "final.html",
    )
    state_log = tmp_path / "full_states_log_2026-05-25.json"
    state_log.write_text(
        json.dumps(
            {
                "sentiment_report": "# Sentiment\n\nMixed.",
                "investment_debate_state": {
                    "bull_history": "## Bull\n\nUptrend.",
                    "bear_history": "## Bear\n\nCrowded.",
                    "judge_decision": "## RM\n\nScale in.",
                },
                "trader_investment_decision": "## Trader\n\nBuy.",
                "risk_debate_state": {"judge_decision": "## PM\n\nOverweight."},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        output_runner,
        "parse_args",
        lambda argv=None: argparse.Namespace(ticker="QQQ", analysis_date="2026-05-25", stack="fmp", port=8765),
    )
    monkeypatch.setattr(output_runner, "resolve_run_request", lambda ticker, analysis_date: ("QQQ", "2026-05-25"))
    monkeypatch.setattr(output_runner, "build_run_slug", lambda *args, **kwargs: "qqq-test-run")
    monkeypatch.setattr(output_runner, "build_paths", lambda run_path: paths)
    monkeypatch.setattr(output_runner, "ensure_http_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(output_runner, "state_log_path", lambda ticker, analysis_date: state_log)
    monkeypatch.setattr(
        output_runner,
        "write_technical_indicators_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-indicators-report",
            "title": "Technical Indicators Report",
            "category": "technical",
            "markdown_path": "technical-indicators-report.md",
            "html_path": "technical-indicators-report.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-test-run/technical-indicators-report.md",
            "html_url": "http://10.17.17.98:8765/qqq-test-run/technical-indicators-report.html",
        },
    )
    monkeypatch.setattr(
        output_runner,
        "write_technical_chart_artifact",
        lambda *args, **kwargs: {
            "slug": "technical-chart",
            "title": "Technical Chart",
            "category": "technical",
            "markdown_path": "technical-chart.md",
            "html_path": "technical-chart.html",
            "markdown_url": "http://10.17.17.98:8765/qqq-test-run/technical-chart.md",
            "html_url": "http://10.17.17.98:8765/qqq-test-run/technical-chart.html",
        },
    )
    def fake_stream(command, cwd, env, out_handle, capture, on_output=None):
        if "run_fmp_mcp_stack.py" in " ".join(command):
            state_log.write_text(
                json.dumps(
                    {
                        "sentiment_report": "# Sentiment\n\nMixed.",
                        "investment_debate_state": {
                            "bull_history": "## Bull\n\nUptrend.",
                            "bear_history": "## Bear\n\nCrowded.",
                            "judge_decision": "## RM\n\nScale in.",
                        },
                        "trader_investment_decision": "## Trader\n\nBuy.",
                        "risk_debate_state": {"judge_decision": "## PM\n\nOverweight."},
                    }
                ),
                encoding="utf-8",
            )
            if on_output:
                on_output()
            mid_run = json.loads(paths.metadata_json.read_text())
            assert any(artifact["slug"] == "bullish-report" for artifact in mid_run["report_artifacts"])
            assert any(artifact["slug"] == "technical-indicators-report" for artifact in mid_run["report_artifacts"])
            assert any(artifact["slug"] == "technical-chart" for artifact in mid_run["report_artifacts"])
            assert "Published reports" in paths.index_html.read_text()
        return 0

    monkeypatch.setattr(output_runner, "stream_command", fake_stream)

    exit_code = output_runner.run([])

    assert exit_code == 0
    metadata = json.loads(paths.metadata_json.read_text())
    assert metadata["status"] == "completed"
    assert metadata["state_log"] == "full_states_log_2026-05-25.json"
    assert any(artifact["slug"] == "bullish-report" for artifact in metadata["report_artifacts"])
    assert any(artifact["slug"] == "technical-indicators-report" for artifact in metadata["report_artifacts"])
    assert any(artifact["slug"] == "technical-chart" for artifact in metadata["report_artifacts"])
    assert (run_dir / "full_states_log_2026-05-25.json").exists()
    assert (run_dir / "bullish-report.md").exists()
    assert (run_dir / "bullish-report.html").exists()
