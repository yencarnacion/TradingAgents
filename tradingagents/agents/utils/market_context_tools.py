from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows import massive


@tool
def get_intraday_bars(
    symbol: Annotated[str, "Ticker symbol of the company"],
    trade_date: Annotated[str, "Trading date in YYYY-MM-DD format"],
    multiplier: Annotated[int, "Bar size multiplier; 5 means 5-minute bars"] = 5,
    timespan: Annotated[str, "Bar timespan, e.g. minute or hour"] = "minute",
    limit: Annotated[int, "Maximum number of bars to return"] = 5000,
) -> str:
    """Retrieve full-session intraday aggregate bars from Massive."""
    return massive.get_intraday_bars(symbol, trade_date, multiplier, timespan, limit)


@tool
def get_session_bars(
    symbol: Annotated[str, "Ticker symbol of the company"],
    trade_date: Annotated[str, "Trading date in YYYY-MM-DD format"],
    session: Annotated[str, "Session name: premarket, regular, postmarket, or afterhours"] = "premarket",
    multiplier: Annotated[int, "Bar size multiplier; 5 means 5-minute bars"] = 5,
    timespan: Annotated[str, "Bar timespan, e.g. minute or hour"] = "minute",
    limit: Annotated[int, "Maximum number of bars to inspect before session slicing"] = 5000,
) -> str:
    """Retrieve a session-specific slice of Massive intraday aggregate bars."""
    return massive.get_session_bars(symbol, trade_date, session, multiplier, timespan, limit)


@tool
def get_ticker_snapshot(
    symbol: Annotated[str, "Ticker symbol of the company"],
) -> str:
    """Retrieve a current stock snapshot from Massive."""
    return massive.get_ticker_snapshot(symbol)


@tool
def get_market_regime(
    curr_date: Annotated[str, "Trading date in YYYY-MM-DD format"],
) -> str:
    """Retrieve market breadth, benchmark snapshots, sector tape, and optional VIX fallback context."""
    return massive.get_market_regime(curr_date)


@tool
def get_last_trade(
    symbol: Annotated[str, "Ticker symbol of the company"],
) -> str:
    """Retrieve the last trade print for a ticker from Massive."""
    return massive.get_last_trade(symbol)


@tool
def get_nbbo_quotes(
    symbol: Annotated[str, "Ticker symbol of the company"],
    limit: Annotated[int, "Maximum number of recent NBBO quote rows to return"] = 20,
) -> str:
    """Retrieve recent NBBO quotes for a ticker from Massive."""
    return massive.get_nbbo_quotes(symbol, limit)


@tool
def get_options_chain(
    symbol: Annotated[str, "Underlying ticker symbol of the company"],
    trade_date: Annotated[str, "Trading date in YYYY-MM-DD format"],
    expiration_date: Annotated[str, "Optional option expiration date in YYYY-MM-DD format"] = "",
    contract_type: Annotated[str, "Optional option type filter: call or put"] = "",
    strike_window: Annotated[int, "Maximum absolute strike distance from spot price to include"] = 5,
    limit: Annotated[int, "Maximum number of contracts to enrich and return"] = 25,
) -> str:
    """Retrieve nearby option contracts plus previous-close and trade-date aggregate bars from Massive."""
    return massive.get_options_chain(
        symbol,
        trade_date,
        expiration_date=expiration_date or None,
        contract_type=contract_type or None,
        strike_window=strike_window,
        limit=limit,
    )
