import copy
import json
from unittest.mock import patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows import fmp


@pytest.mark.unit
class TestFmpMcpDataflows:
    def setup_method(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
        set_config({"fmp_mcp_url": "http://10.17.17.90:8086/mcp"})

    def test_get_stock_data_formats_chart_rows(self):
        payload_rows = [
            {
                "date": "2026-01-08",
                "open": 101.0,
                "high": 102.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 123,
                "change": -0.5,
                "changePercent": -0.49,
                "vwap": 100.2,
            },
            {
                "date": "2026-01-09",
                "open": 100.5,
                "high": 103.0,
                "low": 100.0,
                "close": 102.25,
                "volume": 456,
                "change": 1.75,
                "changePercent": 1.74,
                "vwap": 101.9,
            },
        ]
        with patch.object(fmp, "call_tool", return_value={"result": json.dumps(payload_rows)}) as mocked:
            result = fmp.get_stock_data("AAPL", "2026-01-08", "2026-01-09")

        mocked.assert_called_once()
        assert "# Stock data for AAPL from 2026-01-08 to 2026-01-09" in result
        assert "date,open,high,low,close,volume,change,changePercent,vwap" in result
        assert "2026-01-08,101.0,102.0,99.5,100.5,123,-0.5,-0.49,100.2" in result
        assert "2026-01-09,100.5,103.0,100.0,102.25,456,1.75,1.74,101.9" in result

    def test_get_fundamentals_combines_profile_quote_and_ratios(self):
        def fake_call(url, tool_name, arguments, verify=None):
            if tool_name == "company":
                return {"result": json.dumps([{
                    "companyName": "Apple Inc.",
                    "description": "Consumer electronics.",
                    "exchangeShortName": "NASDAQ",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "country": "US",
                    "ceo": "Tim Cook",
                    "fullTimeEmployees": 164000,
                    "beta": 1.05,
                    "range": "195.07-311.40",
                    "averageVolume": 43882808,
                    "pe": 35.1,
                }])}
            if tool_name == "quote":
                return {"result": json.dumps([{"marketCap": 4535749279920, "price": 308.82}])}
            if tool_name == "statements" and arguments["endpoint"] == "metrics-ratios-ttm":
                return {"result": json.dumps([{
                    "dividendYieldTTM": 0.0034,
                    "priceToBookRatioTTM": 42.6,
                    "returnOnAssetsTTM": 0.33,
                    "returnOnEquityTTM": 1.46,
                    "currentRatioTTM": 1.07,
                    "debtToEquityRatioTTM": 0.79,
                    "netProfitMarginTTM": 0.27,
                    "enterpriseValueTTM": 4584132279920,
                }])}
            if tool_name == "statements" and arguments["endpoint"] == "financial-scores":
                return {"result": json.dumps([{"altmanZScore": 12.89, "piotroskiScore": 9}])}
            raise AssertionError(f"unexpected call: {tool_name} {arguments}")

        with patch.object(fmp, "call_tool", side_effect=fake_call):
            result = fmp.get_fundamentals("AAPL", "2026-01-10")

        assert "Company Fundamentals for AAPL" in result
        assert "Name: Apple Inc." in result
        assert "Price: 308.82" in result
        assert "Market Cap: 4535749279920" in result
        assert "ROE: 1.46" in result
        assert "Piotroski Score: 9" in result
        assert "Requested as-of date: 2026-01-10" in result

    def test_get_indicators_formats_rsi_history(self):
        payload_rows = [
            {"date": "2026-01-09 00:00:00", "rsi": 27.23},
            {"date": "2026-01-08 00:00:00", "rsi": 26.24},
        ]
        with patch.object(fmp, "call_tool", return_value={"result": json.dumps(payload_rows)}):
            result = fmp.get_indicators("AAPL", "rsi", "2026-01-09", 3)

        assert "## rsi values from 2026-01-06 to 2026-01-09:" in result
        assert "2026-01-09: 27.23" in result
        assert "2026-01-08: 26.24" in result
        assert "relative-strength-index" in result
