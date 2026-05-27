import copy
from unittest.mock import patch

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents.analysts import sentiment_analyst
from tradingagents.dataflows.config import set_config


@pytest.mark.unit
def test_prefetch_sentiment_blocks_uses_x_and_skips_reddit():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))
    set_config({"sentiment_x_source": "grok"})

    with (
        patch.object(sentiment_analyst, "get_news") as mocked_news,
        patch.object(sentiment_analyst, "fetch_stocktwits_messages", return_value="stocktwits-block") as mocked_stocktwits,
        patch.object(sentiment_analyst, "get_x_sentiment_report", return_value="x-block") as mocked_x,
    ):
        mocked_news.func.return_value = "news-block"
        blocks = sentiment_analyst._prefetch_sentiment_blocks("SPY", "2026-01-01", "2026-01-08")

    assert blocks == {
        "news_block": "news-block",
        "stocktwits_block": "stocktwits-block",
        "x_block": "x-block",
    }
    mocked_news.func.assert_called_once_with("SPY", "2026-01-01", "2026-01-08")
    mocked_stocktwits.assert_called_once_with("SPY", limit=30)
    mocked_x.assert_called_once_with("SPY", "2026-01-01", "2026-01-08")
    assert not hasattr(sentiment_analyst, "fetch_reddit_posts")


@pytest.mark.unit
def test_build_system_message_references_x_not_reddit():
    message = sentiment_analyst._build_system_message(
        ticker="SPY",
        start_date="2026-01-01",
        end_date="2026-01-08",
        news_block="news-block",
        stocktwits_block="stocktwits-block",
        x_block="x-block",
    )

    assert "<start_of_x>" in message
    assert "x-block" in message
    assert "Reddit" not in message
    assert "<start_of_reddit>" not in message
