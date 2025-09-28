# Market Intelligence System Codebase Prompt

This project produces a super-intelligent market intelligence dashboard that fuses real-time and historical datasets, advanced technical analysis, machine learning, and sentiment overlays to deliver daily buy-side recommendations for the S&P 500 universe.

## High-Level Flow
1. **Configuration (`IntelligenceConfig`)** – Controls lookback windows, concurrency, pre-market analysis windows, machine-learning horizon, and individual factor weights. Instantiation wires rotating logging, output directories, and optional environment-based ticker limits.
2. **Data Acquisition Modules** – `ComprehensiveMarketIntelligence` composes lightweight provider classes for SEC filings, economic calendar events, congressional trades, social sentiment, institutional flows, insider trades, unusual options activity, news sentiment, and sector rotation strength. Each class currently serves deterministic sample payloads while exposing clear integration points for production APIs.
3. **Pre-Market + Screening Pipeline** – `_run_stock_screening` concurrently evaluates the S&P 500 (respecting limits) using Yahoo Finance daily and intraday data. For each ticker it calculates multi-horizon returns, RSI, SMA trend structure, VWAP, OBV, ATR, unusual volume ratios, EWMA volatility, and pre-market gap/vwap statistics. PatternDetector tags cup-and-handle, double bottom/top, head-and-shoulders, and bull-flag regimes.
4. **Machine Learning + Forecasting** – `MLReturnPredictor` engineers features (returns, volatility, RSI, MACD, ATR, OBV) from each ticker’s 180-day history to train a logistic-regression classifier predicting the probability of a positive 5-day forward return. `VolatilityForecaster` supplies an EWMA-based forward volatility estimate that feeds risk weighting.
5. **Scoring & Recommendations** – `_generate_recommendations` aggregates 17 weighted pillars (technical, fundamentals, quality, risk, volume, pattern, ML, pre-market, options, news, sector rotation, volatility, insider, institutional, social, macro, congressional). Additional heuristics derive holding periods, ATR-based exit strategies, risk/reward ratios, and catalyst narratives. Conviction thresholds map composite scores to STRONG BUY → STRONG SELL ratings.
6. **Reporting** – `_create_dashboard_report` compiles summary metrics, source payloads, recommendation tables, top-pick historical snapshots, and backtest statistics. `_create_html_dashboard` renders an interactive Plotly chart, recommendation table with holding/exit guidance, and intelligence widgets. `_save_report` persists JSON, CSV, and HTML artifacts with timestamped filenames.
7. **Execution** – `main()` builds a production-ready configuration, runs the pipeline, prints CLI highlights (including backtest stats), and lists artifact locations.

## Operational Notes
- Caching via `requests_cache` reduces redundant HTTP calls. Pandas copy-on-write mode and aggressive thread pools keep the run fast.
- Placeholder data sources are deterministic so the project runs without external API keys, but method boundaries are ready for real integrations (Alpha Vantage, Polygon, QuiverQuant, etc.).
- New dependencies include `scikit-learn` for the ML classifier and `vaderSentiment` for NLP scoring. Plotly renders interactive charts in the HTML dashboard.
- The system targets daily execution before the opening bell, highlighting pre-market movers, pattern breakouts, and ML-backed setups for rapid discretionary or systematic trading workflows.
