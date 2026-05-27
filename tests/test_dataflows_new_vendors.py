import copy
import unittest
from unittest.mock import patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows import interface
from tradingagents.agents.analysts import sentiment_analyst


@pytest.mark.unit
class InterfaceVendorRoutingTests(unittest.TestCase):
    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
        self._vendor_methods = {
            method: providers.copy() for method, providers in interface.VENDOR_METHODS.items()
        }

    def tearDown(self):
        interface.VENDOR_METHODS.clear()
        interface.VENDOR_METHODS.update(self._vendor_methods)

    def test_vendor_list_includes_new_vendors(self):
        self.assertIn("massive", interface.VENDOR_LIST)
        self.assertIn("perplexity", interface.VENDOR_LIST)
        self.assertIn("grok", interface.VENDOR_LIST)
        self.assertIn("fmp", interface.VENDOR_LIST)

    def test_default_config_prefers_massive_when_supported(self):
        cfg = copy.deepcopy(default_config.DEFAULT_CONFIG)

        self.assertEqual(cfg["data_vendors"]["core_stock_apis"], "massive")
        self.assertEqual(cfg["data_vendors"]["fundamental_data"], "massive")
        self.assertEqual(cfg["data_vendors"]["news_data"], "massive")
        self.assertEqual(cfg["data_vendors"]["technical_indicators"], "massive")
        self.assertEqual(cfg["tool_vendors"]["get_news"], "massive,fmp,grok")
        self.assertEqual(cfg["tool_vendors"]["get_global_news"], "grok,fmp")
        self.assertEqual(cfg["tool_vendors"]["get_insider_transactions"], "fmp")

    def test_routes_stock_data_to_massive_when_configured(self):
        set_config({"tool_vendors": {"get_stock_data": "massive"}})
        interface.VENDOR_METHODS["get_stock_data"]["massive"] = lambda *args, **kwargs: "massive-stock"

        result = interface.route_to_vendor("get_stock_data", "AAPL", "2026-01-01", "2026-01-10")

        self.assertEqual(result, "massive-stock")

    def test_routes_news_to_perplexity_when_configured(self):
        set_config({"tool_vendors": {"get_news": "perplexity"}})
        interface.VENDOR_METHODS["get_news"]["perplexity"] = lambda *args, **kwargs: "perplexity-news"

        result = interface.route_to_vendor("get_news", "AAPL", "2026-01-01", "2026-01-10")

        self.assertEqual(result, "perplexity-news")

    def test_routes_fundamentals_to_fmp_when_configured(self):
        set_config({"tool_vendors": {"get_fundamentals": "fmp"}})
        interface.VENDOR_METHODS["get_fundamentals"]["fmp"] = lambda *args, **kwargs: "fmp-fundamentals"

        result = interface.route_to_vendor("get_fundamentals", "AAPL", "2026-01-10")

        self.assertEqual(result, "fmp-fundamentals")

    def test_default_news_prefers_massive_over_yfinance(self):
        set_config({"tool_vendors": {"get_news": "massive"}})
        interface.VENDOR_METHODS["get_news"]["massive"] = lambda *args, **kwargs: "massive-news"
        interface.VENDOR_METHODS["get_news"]["yfinance"] = lambda *args, **kwargs: "yfinance-news"

        result = interface.route_to_vendor("get_news", "AAPL", "2026-01-01", "2026-01-10")

        self.assertEqual(result, "massive-news")

    def test_missing_primary_vendor_falls_back_to_available_provider(self):
        set_config({"tool_vendors": {"get_global_news": "massive"}})
        interface.VENDOR_METHODS["get_global_news"]["yfinance"] = lambda *args, **kwargs: "fallback-news"

        result = interface.route_to_vendor("get_global_news", "2026-01-10")

        self.assertEqual(result, "fallback-news")

    def test_news_chain_falls_back_from_massive_to_fmp_then_grok(self):
        set_config({"tool_vendors": {"get_news": "massive,fmp,grok"}})
        interface.VENDOR_METHODS["get_news"]["massive"] = lambda *args, **kwargs: "No news found for AAPL between 2026-01-01 and 2026-01-10"
        interface.VENDOR_METHODS["get_news"]["fmp"] = lambda *args, **kwargs: "fmp-news"
        interface.VENDOR_METHODS["get_news"]["grok"] = lambda *args, **kwargs: "grok-news"

        result = interface.route_to_vendor("get_news", "AAPL", "2026-01-01", "2026-01-10")

        self.assertEqual(result, "fmp-news")

    def test_global_news_chain_falls_back_from_grok_error_to_fmp(self):
        set_config({"tool_vendors": {"get_global_news": "grok,fmp"}})
        interface.VENDOR_METHODS["get_global_news"]["grok"] = lambda *args, **kwargs: "Error retrieving Grok global news: timeout"
        interface.VENDOR_METHODS["get_global_news"]["fmp"] = lambda *args, **kwargs: "fmp-global-news"

        result = interface.route_to_vendor("get_global_news", "2026-01-10")

        self.assertEqual(result, "fmp-global-news")


@pytest.mark.unit
class SentimentXSourceTests(unittest.TestCase):
    def setUp(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))

    def test_optional_x_sentiment_uses_grok_by_default(self):
        with patch.object(sentiment_analyst, "get_x_sentiment_report", return_value="x-block") as mocked:
            result = sentiment_analyst._fetch_optional_x_sentiment("AAPL", "2026-01-01", "2026-01-10")

        mocked.assert_called_once_with("AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "x-block")

    def test_optional_x_sentiment_uses_grok_when_enabled(self):
        set_config({"sentiment_x_source": "grok"})
        with patch.object(sentiment_analyst, "get_x_sentiment_report", return_value="x-block") as mocked:
            result = sentiment_analyst._fetch_optional_x_sentiment("AAPL", "2026-01-01", "2026-01-10")

        mocked.assert_called_once_with("AAPL", "2026-01-01", "2026-01-10")
        self.assertEqual(result, "x-block")
