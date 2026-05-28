import copy
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows import massive
from tradingagents.dataflows.config import set_config


@pytest.mark.unit
class TestMassiveMarketContextDataflows:
    def setup_method(self):
        set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
        set_config({"market_data_mcp_url": "https://10.17.17.90:8083/mcp", "mcp_verify_tls": False})

    def test_get_session_bars_uses_full_intraday_feed_and_slices_premarket(self):
        payload = {
            "results": [
                {"t": 1747902000000, "o": 189.5, "h": 190.1, "l": 189.2, "c": 189.9, "v": 1200},  # 2025-05-22 04:20 ET
                {"t": 1747918200000, "o": 191.0, "h": 191.4, "l": 190.8, "c": 191.2, "v": 3400},  # 2025-05-22 08:50 ET
                {"t": 1747931400000, "o": 193.0, "h": 193.2, "l": 192.7, "c": 193.1, "v": 8100},  # 2025-05-22 12:30 ET
            ]
        }

        with patch.object(massive, "_request", return_value=payload) as mocked:
            result = massive.get_session_bars("AAPL", "2025-05-22", session="premarket")

        mocked.assert_called_once()
        assert "Session window (America/New_York): 04:00-09:29" in result
        assert "2025-05-22 04:20:00 ET" in result
        assert "2025-05-22 08:50:00 ET" in result
        assert "2025-05-22 12:30:00 ET" not in result

    def test_get_session_bars_falls_back_to_previous_completed_session_when_requested_day_is_closed(self):
        payloads = {
            "/v2/aggs/ticker/AAPL/range/5/minute/2025-05-24/2025-05-24": {"results": []},
            "/v2/aggs/ticker/AAPL/range/5/minute/2025-05-23/2025-05-23": {
                "results": [
                    {"t": 1747995600000, "o": 189.5, "h": 190.1, "l": 189.2, "c": 189.9, "v": 1200},  # 2025-05-23 05:40 ET
                    {"t": 1748014200000, "o": 191.0, "h": 191.4, "l": 190.8, "c": 191.2, "v": 3400},  # 2025-05-23 10:50 ET
                ]
            },
        }

        def fake_request(path, params=None):
            return payloads[path]

        with patch.object(massive, "_request", side_effect=fake_request):
            result = massive.get_session_bars("AAPL", "2025-05-24", session="premarket")

        assert "Requested trade date: 2025-05-24" in result
        assert "Using latest completed session: 2025-05-23" in result
        assert "2025-05-23 06:20:00 ET" in result
        assert "2025-05-23 10:50:00 ET" not in result

    def test_get_market_regime_summarizes_breadth_benchmarks_and_sector_tape(self):
        grouped_payload = {
            "results": [
                {"T": "AAPL", "o": 100, "c": 102, "v": 1000},
                {"T": "MSFT", "o": 200, "c": 198, "v": 1500},
                {"T": "NVDA", "o": 300, "c": 300, "v": 900},
            ]
        }
        snapshot_payload = {
            "results": [
                {"ticker": "SPY", "session": {"open": 520, "close": 525, "change": 5, "change_percent": 0.96}},
                {"ticker": "QQQ", "session": {"open": 440, "close": 445, "change": 5, "change_percent": 1.14}},
                {"ticker": "IWM", "session": {"open": 205, "close": 202, "change": -3, "change_percent": -1.46}},
                {"ticker": "XLK", "session": {"open": 210, "close": 213, "change": 3, "change_percent": 1.43}},
                {"ticker": "XLF", "session": {"open": 42, "close": 41.5, "change": -0.5, "change_percent": -1.19}},
            ]
        }

        def fake_request(path, params=None):
            if path == "/v2/aggs/grouped/locale/us/market/stocks/2025-05-22":
                return grouped_payload
            if path == "/v2/snapshot/locale/us/markets/stocks/tickers":
                return snapshot_payload
            raise AssertionError(f"unexpected path: {path} params={params}")

        with patch.object(massive, "_request", side_effect=fake_request):
            result = massive.get_market_regime("2025-05-22")

        assert "Market regime for 2025-05-22" in result
        assert "advancers,decliners,unchanged" in result
        assert "1,1,1" in result
        assert "SPY" in result and "QQQ" in result and "IWM" in result
        assert "XLK" in result and "XLF" in result

    def test_get_market_regime_falls_back_to_previous_completed_session_when_requested_day_is_closed(self):
        grouped_payload = {
            "results": [
                {"T": "AAPL", "o": 100, "c": 102, "v": 1000},
                {"T": "MSFT", "o": 200, "c": 198, "v": 1500},
            ]
        }
        snapshot_payload = {
            "results": [
                {"ticker": "SPY", "session": {"open": 520, "close": 525, "change": 5, "change_percent": 0.96}},
                {"ticker": "QQQ", "session": {"open": 440, "close": 445, "change": 5, "change_percent": 1.14}},
            ]
        }

        def fake_request(path, params=None):
            if path == "/v2/aggs/grouped/locale/us/market/stocks/2025-05-24":
                return {"results": []}
            if path == "/v2/aggs/grouped/locale/us/market/stocks/2025-05-23":
                return grouped_payload
            if path == "/v2/snapshot/locale/us/markets/stocks/tickers":
                return snapshot_payload
            raise AssertionError(f"unexpected path: {path} params={params}")

        with patch.object(massive, "_request", side_effect=fake_request):
            result = massive.get_market_regime("2025-05-24", benchmark_symbols=("SPY", "QQQ"))

        assert "Requested trade date: 2025-05-24" in result
        assert "Using latest completed session: 2025-05-23" in result
        assert "advancers,decliners,unchanged" in result
        assert "SPY" in result and "QQQ" in result

    def test_get_options_chain_enriches_near_spot_contracts_with_prev_close_and_daily_bar(self):
        contracts_payload = {
            "results": [
                {
                    "ticker": "O:AAPL250523C00195000",
                    "expiration_date": "2025-05-23",
                    "strike_price": 195,
                    "contract_type": "call",
                },
                {
                    "ticker": "O:AAPL250523C00200000",
                    "expiration_date": "2025-05-23",
                    "strike_price": 200,
                    "contract_type": "call",
                },
                {
                    "ticker": "O:AAPL250523C00240000",
                    "expiration_date": "2025-05-23",
                    "strike_price": 240,
                    "contract_type": "call",
                },
            ]
        }

        def fake_request(path, params=None):
            if path == "/v2/snapshot/locale/us/markets/stocks/tickers/AAPL":
                return {"results": {"ticker": "AAPL", "session": {"close": 198.4}}}
            if path == "/v3/reference/options/contracts":
                return contracts_payload
            if path == "/v2/aggs/ticker/O:AAPL250523C00195000/prev":
                return {"results": [{"c": 4.2, "v": 1200}]}
            if path == "/v2/aggs/ticker/O:AAPL250523C00200000/prev":
                return {"results": [{"c": 2.7, "v": 980}]}
            if path == "/v2/aggs/ticker/O:AAPL250523C00195000/range/1/day/2025-05-22/2025-05-22":
                return {"results": [{"o": 4.1, "h": 4.8, "l": 3.9, "c": 4.6, "v": 1600}]}
            if path == "/v2/aggs/ticker/O:AAPL250523C00200000/range/1/day/2025-05-22/2025-05-22":
                return {"results": [{"o": 2.5, "h": 3.0, "l": 2.3, "c": 2.8, "v": 1300}]}
            raise AssertionError(f"unexpected path: {path} params={params}")

        with patch.object(massive, "_request", side_effect=fake_request):
            result = massive.get_options_chain("AAPL", "2025-05-22", expiration_date="2025-05-23", contract_type="call", strike_window=5, limit=10)

        assert "Options chain for AAPL on 2025-05-22" in result
        assert "spot_price" in result
        assert "O:AAPL250523C00195000" in result
        assert "O:AAPL250523C00200000" in result
        assert "O:AAPL250523C00240000" not in result
        assert "prev_close" in result
        assert "trade_date_close" in result

    def test_get_options_chain_falls_back_to_previous_completed_session_when_requested_day_is_closed(self):
        contracts_payload = {
            "results": [
                {
                    "ticker": "O:AAPL250523C00195000",
                    "expiration_date": "2025-05-23",
                    "strike_price": 195,
                    "contract_type": "call",
                }
            ]
        }

        def fake_request(path, params=None):
            if path == "/v2/snapshot/locale/us/markets/stocks/tickers/AAPL":
                return {"results": {"ticker": "AAPL", "session": {"close": 198.4}}}
            if path == "/v3/reference/options/contracts" and params.get("as_of") == "2025-05-24":
                return {"results": []}
            if path == "/v3/reference/options/contracts" and params.get("as_of") == "2025-05-23":
                return contracts_payload
            if path == "/v2/aggs/ticker/O:AAPL250523C00195000/prev":
                return {"results": [{"c": 4.2, "v": 1200}]}
            if path == "/v2/aggs/ticker/O:AAPL250523C00195000/range/1/day/2025-05-24/2025-05-24":
                return {"results": []}
            if path == "/v2/aggs/ticker/O:AAPL250523C00195000/range/1/day/2025-05-23/2025-05-23":
                return {"results": [{"o": 4.1, "h": 4.8, "l": 3.9, "c": 4.6, "v": 1600}]}
            raise AssertionError(f"unexpected path: {path} params={params}")

        with patch.object(massive, "_request", side_effect=fake_request):
            result = massive.get_options_chain("AAPL", "2025-05-24", expiration_date="2025-05-23", contract_type="call", strike_window=5, limit=10)

        assert "Requested trade date: 2025-05-24" in result
        assert "Using latest completed session: 2025-05-23" in result
        assert "O:AAPL250523C00195000" in result
        assert "trade_date_close" in result

    def test_get_indicators_computes_supported_series_from_massive_ohlcv(self):
        rows = []
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        for offset in range(260):
            trade_date = start + timedelta(days=offset)
            price = 100 + offset
            rows.append(
                {
                    "t": int(trade_date.timestamp() * 1000),
                    "o": float(price),
                    "h": float(price + 2),
                    "l": float(price - 2),
                    "c": float(price + 1),
                    "v": float(1000 + offset * 10),
                    "vw": float(price + 0.5),
                }
            )

        with patch.object(massive, "_request", return_value={"results": rows}):
            result = massive.get_indicators("AAPL", "close_200_sma", "2025-09-17", 3)

        assert "## close_200_sma values from 2025-09-14 to 2025-09-17:" in result
        assert "2025-09-17:" in result
        assert "200 SMA: A long-term trend benchmark." in result

    def test_get_indicators_fetches_enough_calendar_history_for_200_day_sma(self):
        def fake_request(path, params=None):
            assert path == "/v2/aggs/ticker/AAPL/range/1/day/2024-09-17/2025-09-17"
            rows = []
            current = datetime(2024, 9, 17, 17, tzinfo=timezone.utc)
            end = datetime(2025, 9, 17, 17, tzinfo=timezone.utc)
            offset = 0
            while current <= end:
                if current.weekday() < 5:
                    price = 100 + offset
                    rows.append(
                        {
                            "t": int(current.timestamp() * 1000),
                            "o": float(price),
                            "h": float(price + 2),
                            "l": float(price - 2),
                            "c": float(price + 1),
                            "v": float(1000 + offset * 10),
                            "vw": float(price + 0.5),
                        }
                    )
                    offset += 1
                current += timedelta(days=1)
            return {"results": rows}

        with patch.object(massive, "_request", side_effect=fake_request):
            result = massive.get_indicators("AAPL", "close_200_sma", "2025-09-17", 3)

        assert "2025-09-17: N/A" not in result
        assert "2025-09-17:" in result
