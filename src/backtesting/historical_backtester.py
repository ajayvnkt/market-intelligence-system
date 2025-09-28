"""Historical backtesting harness for the Market Intelligence System.

This module allows analysts to replay the intelligence engine on up to a decade of
historical data, benchmark recommendation performance against major indices, and
persist visual/performance artifacts into a dedicated ``backdata`` folder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from loguru import logger

from ..comprehensive_market_intelligence import (
    ComprehensiveMarketIntelligence,
    IntelligenceConfig,
)

DEFAULT_OUTPUT_DIR = Path.cwd() / "backdata"


@dataclass
class BacktestResult:
    """Container for aggregated backtest results."""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: Dict[str, object]
    chart_path: Path


class HistoricalRecommendationBacktester:
    """Run rolling backtests of the recommendation engine against historical data."""

    def __init__(
        self,
        config: IntelligenceConfig,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        index_symbols: Sequence[str] = ("^GSPC", "^NDX"),
    ) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_symbols = tuple(index_symbols)

        logger.add(
            self.output_dir / "historical_backtest.log",
            rotation="20 MB",
            retention="30 days",
        )

        # Reuse the comprehensive intelligence engine helpers without triggering the
        # full real-time orchestration.
        self.engine = ComprehensiveMarketIntelligence(config)

    def run(
        self,
        start_date: datetime,
        end_date: datetime,
        rebalance: str = "M",
        max_recommendations: Optional[int] = None,
        tickers_limit: Optional[int] = None,
    ) -> BacktestResult:
        """Execute the historical replay.

        Parameters
        ----------
        start_date:
            Beginning of the backtest window.
        end_date:
            End of the backtest window (inclusive).
        rebalance:
            Pandas offset alias describing how often to recompute recommendations.
            Defaults to monthly (``"M"``).
        max_recommendations:
            Override for the number of tickers to keep per rebalance. Falls back to
            ``config.max_recommendations``.
        tickers_limit:
            Optional override for the breadth of the investment universe.
        """

        max_recs = max_recommendations or self.config.max_recommendations
        universe = self._build_universe(tickers_limit)

        logger.info(
            "Starting historical backtest from {start} to {end} with {n} tickers",
            start=start_date.date(),
            end=end_date.date(),
            n=len(universe),
        )

        benchmarks = self._fetch_benchmarks(start_date, end_date)
        schedule = self._generate_schedule(start_date, end_date, rebalance, benchmarks.index)

        trades: List[Dict[str, object]] = []
        recommendation_snapshots: List[pd.DataFrame] = []

        for rebalance_date in schedule:
            logger.info("Running snapshot for %s", rebalance_date.date())
            snapshot, histories, forward_data = self._build_snapshot(
                universe, rebalance_date
            )
            if snapshot.empty:
                logger.warning("No data available for %s", rebalance_date.date())
                continue

            intelligence = {"stock_screening": snapshot}
            recs = self.engine._generate_recommendations(intelligence)  # noqa: SLF001
            if recs.empty:
                logger.warning("No qualifying recommendations on %s", rebalance_date.date())
                continue

            recs = recs.head(max_recs)
            recommendation_snapshots.append(recs.assign(rebalance_date=rebalance_date))

            snapshot_trades = self._evaluate_trades(
                recs,
                histories,
                forward_data,
                rebalance_date,
            )
            trades.extend(snapshot_trades)

        if not trades:
            raise RuntimeError("Historical backtest produced no completed trades.")

        trades_df = pd.DataFrame(trades)
        trades_df.sort_values("exit_date", inplace=True)

        equity_curve = self._build_equity_curve(trades_df)
        performance_summary = self._summarize_performance(trades_df, equity_curve)
        benchmark_curves = self._normalize_benchmarks(benchmarks, equity_curve.index)
        if benchmark_curves:
            start = equity_curve.index[0]
            end = equity_curve.index[-1]
            days = max((end - start).days, 1)
            for symbol, series in benchmark_curves.items():
                final = float(series.iloc[-1])
                performance_summary[f"{symbol}_final"] = final
                performance_summary[f"{symbol}_cagr"] = float(final ** (365.25 / days) - 1)

        chart_path = self._render_chart(equity_curve, benchmark_curves)
        summary_path = self.output_dir / "backtest_summary.json"
        trades_path = self.output_dir / "backtest_trades.csv"
        equity_path = self.output_dir / "backtest_equity_curve.csv"
        snapshots_path = self.output_dir / "backtest_recommendations.csv"

        trades_df.to_csv(trades_path, index=False)
        equity_curve.to_csv(equity_path)

        if recommendation_snapshots:
            pd.concat(recommendation_snapshots, ignore_index=True).to_csv(
                snapshots_path, index=False
            )

        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(performance_summary, fh, indent=2, default=str)

        logger.success(
            "Backtest complete. Trades: %d | Final equity: %.2f",
            len(trades_df),
            equity_curve["equity"].iloc[-1],
        )

        return BacktestResult(
            trades=trades_df,
            equity_curve=equity_curve,
            summary=performance_summary,
            chart_path=chart_path,
        )

    # ------------------------------------------------------------------
    # Universe & scheduling helpers
    # ------------------------------------------------------------------
    def _build_universe(self, tickers_limit: Optional[int]) -> List[str]:
        sp500 = self.engine._get_sp500_tickers()  # noqa: SLF001
        nasdaq100 = self._get_nasdaq_100_tickers()
        universe = sorted({*(sp500 or []), *nasdaq100})
        limit = tickers_limit or self.config.tickers_limit
        if limit:
            universe = universe[:limit]
        return universe

    @staticmethod
    def _get_nasdaq_100_tickers() -> List[str]:
        """Fetch the NASDAQ 100 constituents via Wikipedia fallback."""
        urls = [
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            "https://en.wikipedia.org/wiki/NASDAQ-100",
        ]
        for url in urls:
            try:
                tables = pd.read_html(url, match="Ticker")
                if not tables:
                    continue
                tickers = (
                    tables[0]["Ticker"].astype(str).str.replace(".", "-", regex=False).str.upper()
                )
                if len(tickers) >= 50:
                    return tickers.tolist()
            except Exception as exc:  # pragma: no cover - network resilience
                logger.debug("NASDAQ-100 fetch failed from %s: %s", url, exc)
                continue
        return []

    @staticmethod
    def _generate_schedule(
        start: datetime,
        end: datetime,
        rebalance: str,
        trading_calendar: Iterable[pd.Timestamp],
    ) -> List[pd.Timestamp]:
        calendar = pd.Index(sorted(trading_calendar))
        if calendar.empty:
            raise RuntimeError("Benchmark calendar unavailable; cannot schedule rebalances.")

        raw_schedule = pd.date_range(start=start, end=end, freq=rebalance)
        schedule: List[pd.Timestamp] = []
        for ts in raw_schedule:
            mask = calendar[calendar <= ts]
            if mask.empty:
                continue
            schedule.append(mask[-1])
        return schedule

    def _fetch_benchmarks(
        self, start: datetime, end: datetime
    ) -> pd.DataFrame:
        buffer = timedelta(days=45)
        data = yf.download(
            list(self.index_symbols),
            start=start - buffer,
            end=end + buffer,
            auto_adjust=False,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"].dropna(how="all")
        else:
            closes = data[["Close"]].dropna(how="all")
            closes.columns = list(self.index_symbols)
        return closes

    # ------------------------------------------------------------------
    # Snapshot assembly
    # ------------------------------------------------------------------
    def _build_snapshot(
        self,
        tickers: Sequence[str],
        as_of: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        histories: Dict[str, pd.DataFrame] = {}
        forward: Dict[str, pd.DataFrame] = {}
        rows: List[Dict[str, object]] = []

        lookback = self.config.lookback_days
        forward_buffer = max(self.config.hold_period_bounds) + 10

        for symbol in tickers:
            try:
                hist = yf.download(
                    symbol,
                    start=as_of - timedelta(days=lookback * 2),
                    end=as_of + timedelta(days=forward_buffer),
                    progress=False,
                    auto_adjust=False,
                )
            except Exception as exc:  # pragma: no cover - network resilience
                logger.debug("History download failed for %s: %s", symbol, exc)
                continue

            if hist.empty:
                continue

            hist = hist.dropna(subset=["Close"])
            hist.index = pd.to_datetime(hist.index)
            past = hist[hist.index <= as_of].tail(lookback)
            future = hist[hist.index > as_of]
            if past.empty or len(past) < 60 or future.empty:
                continue

            metrics = self.engine._compute_advanced_metrics(past)  # noqa: SLF001
            row = {
                "ticker": symbol,
                "company": symbol,
                "sector": "Unknown",
                "price": float(past["Close"].iloc[-1]),
                "market_cap": np.nan,
                "pe_ratio": np.nan,
                "forward_pe": np.nan,
                "peg_ratio": np.nan,
                "profit_margin": np.nan,
                "revenue_growth": np.nan,
                "return_on_equity": np.nan,
                "beta": np.nan,
                **metrics,
                "pattern_signals": "Backtest",
                "pre_market_change_pct": np.nan,
                "pre_market_volume": np.nan,
            }
            row["holding_period_days"] = self.engine._estimate_holding_period(row)  # noqa: SLF001
            exit_plan = self.engine._determine_exit_strategy(row)  # noqa: SLF001
            row.update(exit_plan)
            row["exit_strategy"] = row.get("exit_plan", "")
            rows.append(row)

            histories[symbol] = past
            forward[symbol] = future

        if not rows:
            return pd.DataFrame(), {}, {}

        snapshot = pd.DataFrame(rows)
        ml_scores = self.engine.ml_predictor.train_and_score(histories)
        vol_forecasts = self.engine.vol_forecaster.forecast_batch(histories)
        snapshot["ml_probability"] = snapshot["ticker"].map(ml_scores).fillna(0.5)
        snapshot["forecast_volatility"] = (
            snapshot["ticker"].map(vol_forecasts).astype(float).replace({0.0: np.nan})
        )
        snapshot["risk_reward_ratio"] = snapshot.apply(
            self.engine._compute_risk_reward, axis=1
        )  # noqa: SLF001
        return snapshot, histories, forward

    # ------------------------------------------------------------------
    # Trade evaluation & reporting
    # ------------------------------------------------------------------
    def _evaluate_trades(
        self,
        recs: pd.DataFrame,
        histories: Dict[str, pd.DataFrame],
        forward: Dict[str, pd.DataFrame],
        as_of: pd.Timestamp,
    ) -> List[Dict[str, object]]:
        trades: List[Dict[str, object]] = []
        for _, rec in recs.iterrows():
            ticker = rec["ticker"]
            past = histories.get(ticker)
            future = forward.get(ticker)
            if past is None or future is None or future.empty:
                continue

            entry_price = float(past["Close"].iloc[-1])
            hold_days = int(rec.get("holding_period_days", self.config.hold_period_bounds[0]))
            future_closes = future["Close"].dropna()
            if len(future_closes) < hold_days:
                continue
            exit_price = float(future_closes.iloc[hold_days - 1])
            ret = (exit_price / entry_price) - 1

            trades.append(
                {
                    "ticker": ticker,
                    "rebalance_date": as_of,
                    "exit_date": future_closes.index[hold_days - 1],
                    "holding_period_days": hold_days,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "return": ret,
                    "conviction_score": float(rec["conviction_score"]),
                    "recommendation": rec["recommendation"],
                }
            )
        return trades

    @staticmethod
    def _build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
        equity = 1.0
        points: List[Tuple[pd.Timestamp, float]] = []
        for _, trade in trades.sort_values("exit_date").iterrows():
            equity *= 1 + float(trade["return"])
            points.append((trade["exit_date"], equity))
        curve = pd.DataFrame(points, columns=["date", "equity"]).set_index("date")
        return curve

    @staticmethod
    def _summarize_performance(trades: pd.DataFrame, curve: pd.DataFrame) -> Dict[str, float]:
        avg_return = float(trades["return"].mean())
        median_return = float(trades["return"].median())
        hit_rate = float((trades["return"] > 0).mean())
        total_trades = int(len(trades))
        start = trades["rebalance_date"].min()
        end = trades["exit_date"].max()
        days = max((end - start).days, 1)
        final_equity = float(curve["equity"].iloc[-1]) if not curve.empty else 1.0
        cagr = (final_equity ** (365.25 / days) - 1) if final_equity > 0 else np.nan

        drawdown = HistoricalRecommendationBacktester._max_drawdown(curve["equity"])

        return {
            "average_return": avg_return,
            "median_return": median_return,
            "hit_rate": hit_rate,
            "total_trades": total_trades,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "final_equity": final_equity,
            "cagr": float(cagr),
            "max_drawdown": float(drawdown),
        }

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        cumulative_max = equity.cummax()
        drawdowns = equity / cumulative_max - 1
        return float(drawdowns.min())

    def _normalize_benchmarks(
        self, benchmarks: pd.DataFrame, window: Iterable[pd.Timestamp]
    ) -> Dict[str, pd.Series]:
        normalized: Dict[str, pd.Series] = {}
        window_idx = pd.Index(sorted(window))
        for symbol in self.index_symbols:
            if symbol not in benchmarks.columns:
                continue
            series = benchmarks[symbol].reindex(window_idx).dropna()
            if series.empty:
                continue
            normalized[symbol] = series / series.iloc[0]
        return normalized

    def _render_chart(
        self,
        equity_curve: pd.DataFrame,
        benchmarks: Dict[str, pd.Series],
    ) -> Path:
        fig = go.Figure()
        strategy_series = equity_curve["equity"]
        if not strategy_series.empty:
            strategy_series = strategy_series / strategy_series.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=strategy_series.index,
                y=strategy_series,
                name="Strategy Equity",
                line=dict(width=3, color="#2E86AB"),
            )
        )
        for symbol, series in benchmarks.items():
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    name=symbol,
                    line=dict(dash="dash"),
                )
            )
        fig.update_layout(
            title="Recommendation Engine Backtest vs Benchmarks",
            xaxis_title="Date",
            yaxis_title="Normalized Performance",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        chart_path = self.output_dir / "backtest_performance.html"
        fig.write_html(chart_path)
        return chart_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a historical backtest of the Market Intelligence recommendations.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.utcnow() - timedelta(days=365 * 10)).strftime("%Y-%m-%d"),
        help="Backtest window start date (YYYY-MM-DD). Defaults to 10 years ago.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="Backtest window end date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--rebalance",
        type=str,
        default="M",
        help="Pandas offset alias for rebalance frequency (e.g., M, W-FRI).",
    )
    parser.add_argument(
        "--tickers-limit",
        type=int,
        default=75,
        help="Limit the number of tickers to evaluate for speed.",
    )
    parser.add_argument(
        "--max-recommendations",
        type=int,
        default=15,
        help="Maximum recommendations to hold per rebalance.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for backtest artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    config = IntelligenceConfig(
        output_dir=Path(args.output_dir),
        tickers_limit=args.tickers_limit,
    )
    config.max_recommendations = args.max_recommendations
    config.ml_min_samples = 150

    backtester = HistoricalRecommendationBacktester(config=config, output_dir=config.output_dir)
    result = backtester.run(
        start,
        end,
        args.rebalance,
        max_recommendations=args.max_recommendations,
        tickers_limit=args.tickers_limit,
    )
    print(json.dumps(result.summary, indent=2))


if __name__ == "__main__":
    main()
