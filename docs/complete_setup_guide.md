# 🚀 Super-Intelligent Market Intelligence – Setup & Operations Guide

This guide walks through environment setup, configuration, and daily operating procedures for the upgraded market intelligence engine.

## 1. Environment Setup
```bash
# Clone and enter the repository
git clone https://github.com/ajayvnkt/market-intelligence-system.git
cd market-intelligence-system

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install core dependencies
pip install -r requirements.txt
```

### Optional Enhancements
- **requests-cache** is enabled automatically when installed to cache HTTP calls.
- Add `plotly>=5.15` for richer offline charts if not already installed.

## 2. API Configuration (Optional)
The project ships with deterministic sample payloads so it runs out-of-the-box. For live data, export API keys before execution:
```bash
export SEC_API_KEY=your_key
export FRED_API_KEY=your_key
export TWITTER_BEARER_TOKEN=your_token
# add ALPHAVANTAGE_API_KEY, POLYGON_API_KEY, etc. as you integrate providers
```

## 3. Quick Start
```bash
python -m src.comprehensive_market_intelligence
```
Outputs are saved to `./market_intelligence`:
- `dashboard_<timestamp>.html` – interactive dashboard (Plotly chart, tables, catalysts)
- `recommendations_<timestamp>.csv` – enriched recommendations with ML probability, holding period, exit plan
- `market_intelligence_<timestamp>.json` – machine-friendly report containing every data source and analytics block
- `intelligence.log` – rotating diagnostics log

## 4. Configuration Highlights
The `IntelligenceConfig` dataclass exposes the primary levers:

| Setting | Purpose | Default |
| --- | --- | --- |
| `lookback_days` | Daily history pulled per ticker | 252 |
| `premarket_window_minutes` | Pre-market window for VWAP/gap analysis | 120 |
| `volume_spike_threshold` | Ratio trigger for unusual-volume scoring | 1.75 |
| `ml_lookahead_days` | Forward horizon for ML return prediction | 5 |
| `hold_period_bounds` | Min/Max days for suggested holding window | (3, 21) |
| `max_workers` | Thread pool size for ticker processing | 12 |
| `min_conviction_score` | Minimum composite score kept in results | 60 |

### Factor Weights (excerpt)
The scoring engine combines 17 subscores. Adjust weights to fit your style:

| Weight | What it captures | Default |
| --- | --- | --- |
| `technical_weight` | RSI, trend, multi-horizon momentum | 0.16 |
| `volume_weight` | Unusual volume, VWAP, OBV | 0.07 |
| `pattern_weight` | Cup & handle, double bottom/top, flags | 0.05 |
| `ml_weight` | Logistic regression probability | 0.12 |
| `premarket_weight` | Gap magnitude and liquidity | 0.05 |
| `options_weight` | Unusual call/put sweeps | 0.06 |
| `volatility_weight` | Forward EWMA/GARCH-lite volatility | 0.05 |
| `insider_weight` | Recent insider buying vs selling | 0.05 |
| `social_weight` | Viral influence score | 0.04 |
| `macro_weight` | Sector bias from macro view | 0.04 |

## 5. Daily Workflow
1. **90 minutes before open** – Run the pipeline to capture pre-market gaps and new filings.
2. **Review dashboard** – Focus on STRONG BUY/BUY names with ML probability > 0.60 and healthy risk/reward (>2.0).
3. **Validate catalysts** – Check options flow, insider activity, and news sentiment to confirm each setup.
4. **Plan exits** – Use the ATR-based exit strategies and holding period guidance included in the table.
5. **Log performance** – Compare new outputs with the embedded backtest hit-rate to monitor regime changes.

### Optional: Historical Validation
```bash
python -m src.backtesting.historical_backtester --start 2014-01-01 --end 2024-01-01 --rebalance M
```
Artifacts are written to `backdata/`:
- `backtest_trades.csv` – individual trades with realized returns
- `backtest_equity_curve.csv` – compounded strategy equity
- `backtest_summary.json` – CAGR, hit rate, drawdown, and totals
- `backtest_performance.html` – Plotly chart comparing S&P 500 and Nasdaq 100 benchmarks

## 6. Extending the System
- **Live Data Sources:** Replace sample DataFrames in `SECFilingMonitor`, `OptionsFlowAnalyzer`, etc. with API integrations (SEC, Polygon, QuiverQuant, Alpha Vantage, NewsAPI, etc.).
- **Alternative Models:** Swap the logistic regression in `MLReturnPredictor` for gradient boosting, random forests, or neural networks by modifying the pipeline.
- **Additional Indicators:** Leverage the metrics computed in `_compute_advanced_metrics` to add ATR bands, MACD crosses, or sector-relative z-scores.
- **Automation:** Schedule via cron, Airflow, or GitHub Actions; archives are timestamped so multiple daily runs coexist.

## 7. Troubleshooting
| Issue | Fix |
| --- | --- |
| Yahoo Finance rate limits | Reduce `max_workers`, enable `requests_cache`, or supply paid data feeds |
| Empty recommendations | Lower `min_conviction_score` or expand `tickers_limit` |
| HTML missing chart | Ensure `plotly` is installed or run `pip install plotly` |
| ML score stuck at 0.50 | Increase `ml_min_samples` or expand `tickers_limit` for more training data |

## 8. Next Steps
- Wire live APIs for pre-market quotes and options flow.
- Persist historical recommendations to evaluate long-term performance.
- Hook output into a Streamlit or FastAPI front-end for intra-day monitoring.

With these upgrades the system behaves like a professional analyst desk—ingesting cross-asset intelligence, running predictive models, and serving clear trading playbooks each morning. Happy hunting! 🧠📈
