# FMP + Massive Expansion Plan

> **For Hermes:** Use `subagent-driven-development` if executing this later.

**Goal:** Expand the current repo so the FMP stack gains higher-signal event/catalyst context while price-heavy, intraday, options, and microstructure analysis move to Massive/Polygon via the existing market-data MCP.

**Architecture:** Keep the current analyst graph intact and add new tools/dataflows behind the existing analyst roles instead of creating new agents in phase 1. Route event/catalyst/fundamental context through FMP where the user’s plan allows it, but route quote-heavy and intraday/options workflows through Massive because the market-data MCP already exposes those endpoints reliably and FMP subscription access is partially restricted.

**Tech Stack:** Python, LangChain tool wrappers, existing `route_to_vendor()` dispatch, FMP MCP (`http://10.17.17.90:8086/mcp`), Massive/Polygon MCP (`https://10.17.17.90:8083/mcp`).

---

## Verified MCP constraints to plan around

### Verified available now
- **FMP quote works** via MCP (`quote-short` returned AAPL quote successfully).
- **FMP analyst estimates work** via MCP (`financial-estimates` returned AAPL estimate rows).
- **FMP ratings snapshot works** via MCP (`ratings-snapshot` returned AAPL score/rating).
- **FMP SEC filings work** via MCP (`search-by-symbol` returned recent AAPL filings when `from_date`/`to_date` were provided).
- **FMP earnings calendar works** via MCP (`earnings-company` returned AAPL earnings dates/estimates).
- **FMP economics indicators work** via MCP (`economics-indicators` for CPI returned history).
- **FMP index quote works** via MCP (`index-quote-short` for `^VIX` worked).
- **FMP sector performance snapshot works** when a market date is supplied.
- **Massive intraday bars work** through the market-data MCP (`/v2/aggs/.../5/minute/...`).
- **Massive full-session minute bars include premarket and after-hours**. AAPL minute aggregates for `2026-05-22` started at `2026-05-22T08:00:00Z` (4:00am ET) and extended to `2026-05-22T23:59:00Z` (7:59pm ET), so one aggregate feed can power premarket / regular / postmarket segmentation.
- **Massive stock snapshot works** through the market-data MCP (`/v2/snapshot/locale/us/markets/stocks/tickers/AAPL`).
- **Massive multi-ticker snapshot works** through the market-data MCP (`/v2/snapshot/locale/us/markets/stocks/tickers?tickers=...`) and can cover SPY/QQQ/IWM plus sector ETFs in one call.
- **Massive options contracts work** through the market-data MCP (`/v3/reference/options/contracts`).
- **Massive grouped daily stock aggregates work** through the market-data MCP (`/v2/aggs/grouped/locale/us/market/stocks/{date}`) and returned a full US stock universe sample large enough to compute advance/decline breadth.
- **Massive last trade + quotes work** through the market-data MCP (`/v2/last/trade/AAPL`, `/v3/quotes/AAPL`).
- **Massive previous-close and option aggregate bars work** through the market-data MCP (`/v2/aggs/ticker/{symbol}/prev`, `/v2/aggs/ticker/{option_contract}/range/...`).

### Verified restricted / plan-sensitive now
- **FMP earnings transcripts are restricted on the current subscription** (402 from `transcripts-dates-by-symbol`).
- **FMP ETF holdings are restricted on the current subscription** (402 from `etf/holdings`).
- **Massive live option snapshot / live option quotes / live option last-trade endpoints are not authorized on the current subscription** (`/v3/snapshot/options/...`, `/v3/quotes/{option_contract}`, `/v2/last/trade/{option_contract}` returned auth errors).

### Planning implication
1. **Do not build phase-1 quote/intraday/options logic on FMP.** Use Massive.
2. **Do build FMP estimates/ratings/filings/calendars/economics/sector context.** These are already accessible.
3. **Implement transcript + ETF-holdings code paths as optional capability-gated features** that fail soft with a clear “subscription-restricted” message instead of breaking the analyst.
4. **Implement Massive options analysis as reference + aggregate-price-action based in phase 1**, not as full live-greeks/open-interest analysis, unless the Massive entitlement is upgraded.

---

## Phase-1 product decisions

### Keep the existing analyst roles
Do **not** add new analysts yet. Feed the new data into the existing reports:
- **Market Analyst** gets intraday, pre/post-market, options, and regime context.
- **Fundamentals Analyst** gets estimates, ratings, filings, transcripts, ETF holdings.
- **News Analyst** gets earnings calendar, economics calendar, macro/index/sector context, and filing/event context.
- **Sentiment Analyst** stays focused on news + StockTwits + Reddit + X; do not overload it in phase 1.

### Minimize graph churn
Do **not** add new `AgentState` report fields in phase 1 unless the reports become too large. Reuse:
- `market_report`
- `fundamentals_report`
- `news_report`

This keeps downstream researcher/risk/trader prompts unchanged because they already ingest these reports.

---

## API-to-analyst mapping

| New data | Backing API/MCP | Analyst prompt that should consume it | Why |
|---|---|---|---|
| Earnings transcripts | FMP MCP if available, else soft-restricted | `agents/analysts/fundamentals_analyst.py` | Best place for management tone, guidance, business segment commentary |
| Analyst estimates / ratings | FMP MCP | `agents/analysts/fundamentals_analyst.py` | Improves expectation-vs-reality view |
| SEC filings | FMP MCP | `agents/analysts/fundamentals_analyst.py` and `agents/analysts/news_analyst.py` | Fundamentals uses the facts; news uses recent filing events/catalysts |
| Earnings calendar | FMP MCP | `agents/analysts/news_analyst.py` | Event risk / upcoming catalyst timing |
| Economics calendar + indicators | FMP MCP | `agents/analysts/news_analyst.py` | Macro catalyst timing |
| Sector / index / VIX context | FMP MCP | `agents/analysts/news_analyst.py` and `agents/analysts/market_analyst.py` | News frames macro tape; market analyst uses regime/timing |
| ETF holdings | FMP MCP if available, else soft-restricted | `agents/analysts/fundamentals_analyst.py` | Useful for sector-flow names; do not block if unavailable |
| Intraday bars | Massive MCP | `agents/analysts/market_analyst.py` | Highest value for daytrade timing |
| Pre/post-market bars | Massive MCP | `agents/analysts/market_analyst.py` | Gap and session-transition context |
| Options chain / contract reference + aggregate-price-action | Massive MCP | `agents/analysts/market_analyst.py` | Near-term strike map and contract price action; good enough for phase 1 without promising greeks/OI |
| Market regime block (SPY/QQQ/IWM + sector breadth + VIX) | Massive MCP + FMP VIX/sector snapshot fallback | `agents/analysts/market_analyst.py` and `agents/analysts/news_analyst.py` | Separates single-name signal from tape/regime |
| Trade / quote granularity | Massive MCP | `agents/analysts/market_analyst.py` | Optional later enhancement for tape-style reads |

---

## Exact files to modify

### Existing files to extend
- `tradingagents/default_config.py`
- `tradingagents/dataflows/interface.py`
- `tradingagents/dataflows/fmp.py`
- `tradingagents/dataflows/massive.py`
- `tradingagents/agents/utils/agent_utils.py`
- `tradingagents/agents/utils/fundamental_data_tools.py`
- `tradingagents/agents/utils/news_data_tools.py`
- `tradingagents/agents/utils/core_stock_tools.py`
- `tradingagents/agents/analysts/fundamentals_analyst.py`
- `tradingagents/agents/analysts/news_analyst.py`
- `tradingagents/agents/analysts/market_analyst.py`
- `tests/test_dataflows_fmp.py`
- `tests/test_dataflows_new_vendors.py`

### New files to add
- `tradingagents/agents/utils/regime_tools.py`
- `tests/test_dataflows_massive.py`
- `tests/test_market_regime_tools.py`

---

## New adapters / functions to add

### FMP adapter additions in `tradingagents/dataflows/fmp.py`
Add these functions:
- `get_earnings_transcripts(ticker: str, limit: int = 3)`
- `get_analyst_estimates(ticker: str, period: str = "annual")`
- `get_analyst_ratings(ticker: str)`
- `get_sec_filings(ticker: str, from_date: str, to_date: str, limit: int = 10)`
- `get_earnings_calendar(ticker: str, limit: int = 6)`
- `get_economics_calendar(from_date: str, to_date: str, country: str | None = None)`
- `get_economic_indicator(name: str, limit: int = 12)`
- `get_sector_context(curr_date: str)`
- `get_index_context(symbols: list[str] | None = None)`
- `get_etf_holdings(symbol: str, limit: int = 15)`

Implementation notes:
- Add a helper such as `_restricted_message(feature, error)` that returns a report-friendly restriction string.
- Catch `MCPToolError`/402-like restricted responses for transcripts and ETF holdings and return a non-fatal block like:
  - `FMP transcript access is restricted on the current subscription; skipping transcript analysis.`
- Do **not** use FMP quote expansion for phase 1 beyond current fundamentals use.

### Massive adapter additions in `tradingagents/dataflows/massive.py`
Add these functions:
- `get_intraday_bars(symbol: str, trade_date: str, multiplier: int = 5, timespan: str = "minute", limit: int = 120)`
- `get_session_bars(symbol: str, trade_date: str, session: str = "premarket")`
- `get_ticker_snapshot(symbol: str)`
- `get_options_chain(symbol: str, trade_date: str, expiration_date: str | None = None, contract_type: str | None = None, strike_window: int = 5, limit: int = 25)`
- `get_market_regime(curr_date: str, benchmark_symbols: tuple[str, ...] = ("SPY", "QQQ", "IWM"))`
- `get_last_trade(symbol: str)`
- `get_nbbo_quotes(symbol: str, limit: int = 20)`

Implementation notes:
- Reuse `_request()` and the MCP `call_api` path already in the file.
- Normalize CSV/JSON responses into human-readable CSV blocks just like current functions.
- Implement `get_session_bars()` by calling the same full-day intraday aggregate feed and slicing timestamps into ET sessions instead of assuming a separate premarket endpoint exists.
- Implement `get_options_chain()` as a **two-step enrichment**: (1) fetch contract references from `/v3/reference/options/contracts`; (2) choose a small strike band around spot and fetch aggregate bars / previous-close for those contracts. This yields a practical options context block even without live option snapshot entitlement.
- `get_market_regime()` should gather:
  - SPY / QQQ / IWM daily + intraday snapshot
  - market breadth from Massive grouped-daily stock aggregates (`advance/decline/flat`, optional up-volume vs down-volume)
  - sector tape from sector ETF snapshots (e.g. XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLU)
  - VIX quote (FMP index quote is acceptable fallback)
- Make session functions explicitly report the session window they cover so the prompt can reason about premarket vs regular vs postmarket behavior.

---

## Tool-wrapper layer to add

### `tradingagents/agents/utils/fundamental_data_tools.py`
Add LangChain `@tool` wrappers for:
- `get_earnings_transcripts`
- `get_analyst_estimates`
- `get_analyst_ratings`
- `get_sec_filings`
- `get_etf_holdings`

Each should dispatch through `route_to_vendor(...)` where appropriate. For transcript/ETF-holdings, it is fine if only `fmp` implements them initially.

### `tradingagents/agents/utils/news_data_tools.py`
Add wrappers for:
- `get_earnings_calendar`
- `get_economics_calendar`
- `get_economic_indicator`
- `get_sector_context`
- `get_index_context`

### `tradingagents/agents/utils/core_stock_tools.py`
Add wrappers for:
- `get_intraday_bars`
- `get_session_bars`
- `get_ticker_snapshot`
- `get_options_chain`
- `get_last_trade`
- `get_nbbo_quotes`

### New file `tradingagents/agents/utils/regime_tools.py`
Create a dedicated wrapper:
- `get_market_regime(curr_date: str)`

Rationale: keep the market-regime prompt input as one coherent tool call rather than forcing the LLM to stitch together 5+ low-level calls.

### `tradingagents/agents/utils/agent_utils.py`
Import and re-export the new tool wrappers so analyst files can bind them.

---

## Routing changes in `tradingagents/dataflows/interface.py`

### Extend `TOOLS_CATEGORIES`
Add categories or extend existing ones:
- `core_stock_apis`: include intraday/session/snapshot/last_trade/quotes/options/regime tools
- `fundamental_data`: include estimates/ratings/transcripts/ETF holdings
- `news_data`: include filings/calendars/economics/sector/index context

### Extend `VENDOR_METHODS`
Add mappings such as:
- `get_intraday_bars` → `massive`
- `get_session_bars` → `massive`
- `get_ticker_snapshot` → `massive`
- `get_options_chain` → `massive`
- `get_last_trade` → `massive`
- `get_nbbo_quotes` → `massive`
- `get_market_regime` → `massive` (with FMP fallback inside implementation if needed)
- `get_earnings_transcripts` → `fmp`
- `get_analyst_estimates` → `fmp`
- `get_analyst_ratings` → `fmp`
- `get_sec_filings` → `fmp`
- `get_earnings_calendar` → `fmp`
- `get_economics_calendar` → `fmp`
- `get_economic_indicator` → `fmp`
- `get_sector_context` → `fmp`
- `get_index_context` → `fmp`
- `get_etf_holdings` → `fmp`

### Fallback behavior
Do not rely on `route_to_vendor()` alone for plan/subscription limitations because it only auto-falls back on `AlphaVantageRateLimitError`. For FMP subscription restrictions, handle the fallback/soft-fail inside `fmp.py` itself for phase 1.

---

## Analyst prompt changes

### 1. `tradingagents/agents/analysts/fundamentals_analyst.py`
Expand the tool list to include:
- `get_earnings_transcripts`
- `get_analyst_estimates`
- `get_analyst_ratings`
- `get_sec_filings`
- `get_etf_holdings`

Prompt updates:
- Tell the analyst to compare valuation/fundamentals against **forward estimates**.
- Tell it to look for **management guidance / tone** from transcripts if available.
- Tell it to summarize **material SEC filings** and changes in disclosure.
- Tell it to use ETF-holdings context only when relevant to sector-flow names and to ignore it when unavailable.

### 2. `tradingagents/agents/analysts/news_analyst.py`
Expand the tool list to include:
- `get_earnings_calendar`
- `get_economics_calendar`
- `get_economic_indicator`
- `get_sec_filings`
- `get_sector_context`
- `get_index_context`

Prompt updates:
- Require an **upcoming catalyst section**.
- Require a **macro calendar risk section**.
- Require the analyst to distinguish **single-name news** from **market-regime / sector tape**.
- Require explicit mention of whether the stock is moving with its sector/index or idiosyncratically.

### 3. `tradingagents/agents/analysts/market_analyst.py`
Expand the tool list to include:
- `get_market_regime`
- `get_intraday_bars`
- `get_session_bars`
- `get_ticker_snapshot`
- `get_options_chain`
- `get_last_trade`
- `get_nbbo_quotes`

Prompt updates:
- Make `get_market_regime()` the **first required call**.
- Require an **intraday structure section**:
  - opening move
  - midday trend/chop
  - late-day behavior
  - pre/post-market positioning
- Require an **options positioning section** based on near-dated strikes, contract selection, and aggregate contract price action; do **not** ask for greeks/open-interest unless those fields are actually present.
- Add guidance to use last trade / NBBO only when making a short-horizon execution/timing read, not as a replacement for the broader intraday summary.

### 4. `tradingagents/agents/analysts/sentiment_analyst.py`
No phase-1 API changes. Keep focused.

---

## Config changes in `tradingagents/default_config.py`
Add these optional keys:
- `intraday_bar_limit`
- `intraday_bar_multiplier`
- `intraday_bar_timespan`
- `options_chain_limit`
- `regime_benchmark_tickers` (default `['SPY', 'QQQ', 'IWM']`)
- `macro_calendar_countries` (default `['US']`)
- `filings_lookback_days`
- `transcript_limit`

Also add env overrides only if you want shell-level control for these. Not mandatory in phase 1.

---

## Test plan

### Update `tests/test_dataflows_fmp.py`
Add tests for:
- estimates formatting
- ratings formatting
- SEC filings formatting
- earnings calendar formatting
- economics indicator formatting
- restricted transcript response handling
- restricted ETF-holdings response handling

### Create `tests/test_dataflows_massive.py`
Add tests for:
- intraday bar formatting
- session-bar formatting
- options-chain formatting
- snapshot formatting
- last-trade formatting
- NBBO quote formatting
- market-regime aggregation formatting

### Update `tests/test_dataflows_new_vendors.py`
Add routing tests for the new methods, especially:
- `get_intraday_bars` routed to `massive`
- `get_options_chain` routed to `massive`
- `get_analyst_estimates` routed to `fmp`
- `get_sec_filings` routed to `fmp`

### Create `tests/test_market_regime_tools.py`
Add tool-wrapper tests ensuring `get_market_regime()` and the new wrappers call `route_to_vendor()` with the expected method names and args.

---

## Bite-sized execution order

### Task 1: Extend FMP dataflow with non-restricted phase-1 endpoints
**Files:**
- Modify: `tradingagents/dataflows/fmp.py`
- Test: `tests/test_dataflows_fmp.py`

Implement estimates, ratings, SEC filings, earnings calendar, economics indicators, sector/index context.

### Task 2: Add FMP soft-fail handling for restricted endpoints
**Files:**
- Modify: `tradingagents/dataflows/fmp.py`
- Test: `tests/test_dataflows_fmp.py`

Implement transcript and ETF-holdings functions that return non-fatal restriction messages when 402-like restrictions occur.

### Task 3: Extend Massive dataflow for intraday and options
**Files:**
- Modify: `tradingagents/dataflows/massive.py`
- Create: `tests/test_dataflows_massive.py`

Implement intraday bars, session bars, ticker snapshot, options chain, last trade, NBBO quotes.

### Task 4: Add market-regime aggregation tool
**Files:**
- Modify: `tradingagents/dataflows/massive.py`
- Create: `tradingagents/agents/utils/regime_tools.py`
- Create: `tests/test_market_regime_tools.py`

Implement one high-level regime block for SPY/QQQ/IWM + sector breadth + VIX.

### Task 5: Extend routing and wrappers
**Files:**
- Modify: `tradingagents/dataflows/interface.py`
- Modify: `tradingagents/agents/utils/fundamental_data_tools.py`
- Modify: `tradingagents/agents/utils/news_data_tools.py`
- Modify: `tradingagents/agents/utils/core_stock_tools.py`
- Modify: `tradingagents/agents/utils/agent_utils.py`
- Update tests: `tests/test_dataflows_new_vendors.py`, `tests/test_market_regime_tools.py`

### Task 6: Update analyst prompts
**Files:**
- Modify: `tradingagents/agents/analysts/fundamentals_analyst.py`
- Modify: `tradingagents/agents/analysts/news_analyst.py`
- Modify: `tradingagents/agents/analysts/market_analyst.py`

### Task 7: Add config defaults
**Files:**
- Modify: `tradingagents/default_config.py`

### Task 8: Run focused tests
Run:
- `source .venv/bin/activate && pytest tests/test_dataflows_fmp.py -q`
- `source .venv/bin/activate && pytest tests/test_dataflows_massive.py -q`
- `source .venv/bin/activate && pytest tests/test_dataflows_new_vendors.py -q`
- `source .venv/bin/activate && pytest tests/test_market_regime_tools.py -q`

### Task 9: Run broader regression slice
Run:
- `source .venv/bin/activate && pytest tests/test_analyst_execution.py -q`
- `source .venv/bin/activate && pytest tests/test_structured_agents.py -q`

---

## Final recommendation

### What to implement first
1. **Massive intraday + market regime + options**
2. **FMP estimates + ratings + filings + calendars**
3. **FMP transcripts + ETF holdings as capability-gated enrichments**

### Why this order
- It matches the verified MCP access you actually have today.
- It avoids building phase-1 features around endpoints currently blocked by subscription.
- It gives the biggest daytrading improvement first: timing, regime, options, execution context.
- It still improves catalyst analysis materially through FMP where access is already present.

### Revisited conclusion after deeper Massive checks
- **Yes, Massive can provide the core market-analysis information we want the analyst to get**: full-session minute bars (including pre/post), benchmark/index ETF snapshots, broad advance/decline breadth, last trade, and NBBO quotes.
- **Massive can provide a useful but not fully institutional options view on the current entitlement**: option contract discovery plus contract aggregate bars / previous close work, but live option snapshot/quote/trade endpoints are not currently authorized.
- Therefore the market analyst should be designed to produce:
  - strong intraday structure analysis,
  - strong market-regime analysis,
  - strong tape/execution context using last trade + NBBO,
  - **partial options context** unless the Massive plan is upgraded.

### Explicit planning conclusion on your note
Your instinct is correct: **for quotes/intraday/options, Massive is the better primary path in your current setup.** The MCP checks confirmed Massive already exposes the exact price-heavy endpoints you want, while FMP access is partly restricted for some of the high-value enrichments you mentioned.
