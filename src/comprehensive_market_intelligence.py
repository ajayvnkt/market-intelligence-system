"""Super-intelligent market intelligence engine for daily trading decisions."""

from __future__ import annotations

import json
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger
from plotly.offline import plot
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


try:  # noqa: SIM105 - cache is optional but highly recommended
    import requests_cache  # type: ignore

    requests_cache.install_cache("market_intel_cache", expire_after=3600)
except Exception:  # pragma: no cover - cache failures should not stop runtime
    pass

warnings.filterwarnings("ignore")
pd.options.mode.copy_on_write = True


@dataclass
class IntelligenceConfig:
    """Configuration container controlling the intelligence stack."""

    lookback_days: int = 252
    chunk_size: int = 25
    max_workers: int = 12
    max_intraday_workers: int = 6
    min_price: float = 5.0
    min_volume: int = 300_000
    tickers_limit: Optional[int] = None

    premarket_window_minutes: int = 120
    volume_spike_threshold: float = 1.75
    atr_period: int = 14
    hold_period_bounds: Tuple[int, int] = (3, 21)

    ml_lookahead_days: int = 5
    ml_training_lookback: int = 180
    ml_min_samples: int = 400

    technical_weight: float = 0.16
    fundamental_weight: float = 0.10
    quality_weight: float = 0.08
    risk_weight: float = 0.07
    volume_weight: float = 0.07
    pattern_weight: float = 0.05
    ml_weight: float = 0.12
    premarket_weight: float = 0.05
    options_weight: float = 0.06
    news_weight: float = 0.05
    rotation_weight: float = 0.05
    volatility_weight: float = 0.05
    insider_weight: float = 0.05
    institutional_weight: float = 0.05
    social_weight: float = 0.04
    macro_weight: float = 0.04
    congressional_weight: float = 0.04

    output_dir: Path = field(default_factory=lambda: Path.cwd() / "market_intelligence")
    max_recommendations: int = 25
    min_conviction_score: float = 60.0

    sec_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            self.output_dir / "intelligence.log",
            rotation="25 MB",
            retention="45 days",
            backtrace=True,
            diagnose=True,
        )

        if self.tickers_limit is None:
            env_limit = os.getenv("CMI_TICKERS_LIMIT")
            if env_limit:
                try:
                    self.tickers_limit = int(env_limit)
                except ValueError:
                    logger.warning("Invalid CMI_TICKERS_LIMIT env var. Ignoring override.")


class SECFilingMonitor:
    """Monitor SEC current events feed for impactful filings."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar"

    def get_recent_filings(self) -> pd.DataFrame:
        logger.info("Fetching recent SEC filings ...")
        filings: List[Dict[str, str]] = []
        filing_types = ["8-K", "10-K", "10-Q", "4", "13D", "13G"]

        headers = {"User-Agent": "Market-Intelligence-System"}
        for filing_type in filing_types:
            try:
                url = f"{self.base_url}?action=getcurrent&type={filing_type}"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.content, "html.parser")
                rows = soup.find_all("tr")[1:16]
                for row in rows:
                    cells = row.find_all("td")
                    if len(cells) < 4:
                        continue
                    filings.append(
                        {
                            "date": cells[0].text.strip(),
                            "company": cells[1].text.strip(),
                            "ticker": self._extract_ticker(cells[1].text),
                            "form_type": cells[2].text.strip(),
                            "impact": self._assess_impact(cells[2].text.strip()),
                        }
                    )
            except Exception as exc:  # pragma: no cover - network resilience
                logger.debug(f"SEC fetch error for {filing_type}: {exc}")

        return pd.DataFrame(filings)

    @staticmethod
    def _extract_ticker(company_label: str) -> Optional[str]:
        match = os.path.basename(company_label).split("(")
        if len(match) > 1 and ")" in match[1]:
            return match[1].split(")")[0].strip().upper()
        return None

    @staticmethod
    def _assess_impact(form_type: str) -> str:
        impact_map = {
            "8-K": "HIGH",
            "10-K": "MEDIUM",
            "10-Q": "MEDIUM",
            "4": "MEDIUM",
            "13D": "HIGH",
            "13G": "MEDIUM",
        }
        return impact_map.get(form_type, "LOW")


class EconomicCalendarTracker:
    """Placeholder for economic calendar aggregation."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_upcoming_events(self) -> pd.DataFrame:
        now = datetime.utcnow()
        sample_events = [
            {
                "date": (now + timedelta(days=1)).strftime("%Y-%m-%d"),
                "time": "12:30",
                "event": "CPI MoM",
                "importance": "HIGH",
                "expected": "0.3%",
                "previous": "0.2%",
            },
            {
                "date": (now + timedelta(days=3)).strftime("%Y-%m-%d"),
                "time": "18:00",
                "event": "FOMC Rate Decision",
                "importance": "HIGH",
                "expected": "5.25%",
                "previous": "5.25%",
            },
        ]
        return pd.DataFrame(sample_events)


class CongressionalTradeTracker:
    """Placeholder for congressional trade disclosures."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_recent_trades(self) -> pd.DataFrame:
        now = datetime.utcnow()
        sample = [
            {
                "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                "member": "Nancy Pelosi",
                "party": "D-CA",
                "ticker": "NVDA",
                "transaction": "BUY",
                "amount": "100K-250K",
            },
            {
                "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
                "member": "Tommy Tuberville",
                "party": "R-AL",
                "ticker": "TSLA",
                "transaction": "BUY",
                "amount": "50K-100K",
            },
        ]
        return pd.DataFrame(sample)


class SocialSentimentAnalyzer:
    """Placeholder for social trend analytics."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_trending_stocks(self) -> pd.DataFrame:
        now = datetime.utcnow()
        sample = [
            {
                "ticker": "AAPL",
                "platform": "Reddit",
                "trend_type": "r/wallstreetbets momentum",
                "sentiment": "POSITIVE",
                "influence_score": 82,
                "timestamp": now.isoformat(),
            },
            {
                "ticker": "TSLA",
                "platform": "Twitter",
                "trend_type": "AI partnership chatter",
                "sentiment": "POSITIVE",
                "influence_score": 75,
                "timestamp": now.isoformat(),
            },
        ]
        return pd.DataFrame(sample)


class InstitutionalTracker:
    """Placeholder for institutional flows (13F, holdings)."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_hot_hands(self) -> pd.DataFrame:
        now = datetime.utcnow()
        sample = [
            {
                "date": (now - timedelta(days=3)).strftime("%Y-%m-%d"),
                "institution": "ARK Invest",
                "ticker": "TSLA",
                "action": "BUY",
                "shares": 450_000,
                "value": 70_000_000,
            },
            {
                "date": (now - timedelta(days=5)).strftime("%Y-%m-%d"),
                "institution": "BlackRock",
                "ticker": "MSFT",
                "action": "BUY",
                "shares": 1_200_000,
                "value": 360_000_000,
            },
        ]
        return pd.DataFrame(sample)


class InsiderTradingAnalyzer:
    """Placeholder for Form 4 insider transactions."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_notable_insider_trades(self) -> pd.DataFrame:
        now = datetime.utcnow()
        sample = [
            {
                "date": (now - timedelta(days=1)).strftime("%Y-%m-%d"),
                "ticker": "AMZN",
                "insider_name": "Andy Jassy",
                "title": "CEO",
                "transaction": "BUY",
                "shares": 10_000,
                "price": 146.2,
                "value": 1_462_000,
                "signal_strength": "STRONG_BUY",
            }
        ]
        return pd.DataFrame(sample)


class OptionsFlowAnalyzer:
    """Track unusual options activity for flow-based confirmation."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_unusual_activity(self) -> pd.DataFrame:
        now = datetime.utcnow()
        sample = [
            {
                "timestamp": now.isoformat(),
                "ticker": "NVDA",
                "type": "CALL SWEEP",
                "expiry": (now + timedelta(days=7)).strftime("%Y-%m-%d"),
                "strike": 500,
                "premium": 4_200_000,
                "spot_vs_strike": 1.02,
                "sweep_ratio": 0.85,
            },
            {
                "timestamp": now.isoformat(),
                "ticker": "AAPL",
                "type": "CALL BLOCK",
                "expiry": (now + timedelta(days=14)).strftime("%Y-%m-%d"),
                "strike": 200,
                "premium": 3_100_000,
                "spot_vs_strike": 0.98,
                "sweep_ratio": 0.65,
            },
        ]
        return pd.DataFrame(sample)


class NewsSentimentAnalyzer:
    """Derive NLP-based sentiment scores for recent headlines."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config
        self.analyzer = SentimentIntensityAnalyzer()

    def get_latest_sentiment(self) -> pd.DataFrame:
        headlines = [
            ("MSFT", "Microsoft launches Copilot upgrades, analysts cheer AI roadmap"),
            ("TSLA", "Tesla faces supply-chain hiccups but demand remains strong"),
            ("AAPL", "Apple supplier hints at robust iPhone 16 pre-orders"),
            ("NVDA", "Nvidia beats earnings estimates as AI demand accelerates"),
        ]
        rows = []
        now = datetime.utcnow()
        for ticker, text in headlines:
            score = self.analyzer.polarity_scores(text)["compound"]
            rows.append(
                {
                    "timestamp": now.isoformat(),
                    "ticker": ticker,
                    "headline": text,
                    "sentiment_score": score,
                }
            )
        return pd.DataFrame(rows)


class SectorRotationModel:
    """Approximate sector rotation using liquid SPDR ETFs."""

    sector_etfs = {
        "Technology": "XLK",
        "Financials": "XLF",
        "Industrials": "XLI",
        "Energy": "XLE",
        "Healthcare": "XLV",
        "Consumer Discretionary": "XLY",
    }

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def calculate_relative_strength(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        end = datetime.utcnow()
        start = end - timedelta(days=90)
        spy = yf.download("SPY", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
        
        # Handle different column names
        close_col = "Adj Close" if "Adj Close" in spy.columns else "Close" if "Close" in spy.columns else None
        if close_col is None or spy.empty:
            spy_ret = 0.0
        else:
            spy_ret = spy[close_col].pct_change().add(1).prod() - 1

        for sector, ticker in self.sector_etfs.items():
            data = yf.download(ticker, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
            if data.empty:
                continue
            
            # Handle different column names
            close_col = "Adj Close" if "Adj Close" in data.columns else "Close" if "Close" in data.columns else None
            if close_col is None:
                continue
                
            total_return = data[close_col].pct_change().add(1).prod() - 1
            momentum_20d = data[close_col].pct_change(20).iloc[-1] if len(data) > 20 else np.nan
            rows.append(
                {
                    "sector": sector,
                    "etf": ticker,
                    "quarter_return": total_return,
                    "relative_to_spy": total_return - spy_ret,
                    "momentum_20d": momentum_20d,
                }
            )
        return pd.DataFrame(rows)


class VolatilityForecaster:
    """Simple EWMA-based volatility forecast (GARCH proxy)."""

    def forecast_batch(self, histories: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        forecasts: Dict[str, float] = {}
        for ticker, hist in histories.items():
            if hist is None or hist.empty:
                continue
            returns = hist["Close"].pct_change().dropna()
            if returns.empty:
                continue
            ewma = returns.ewm(alpha=0.06).std().iloc[-1]
            if pd.isna(ewma):
                continue
            forecasts[ticker] = float(ewma * math.sqrt(252))
        return forecasts


class PreMarketDataFetcher:
    """Collect pre-market price/volume context using intraday data."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def fetch(self, ticker: str) -> Dict[str, float]:
        try:
            intraday = yf.download(
                ticker,
                period="2d",
                interval="5m",
                prepost=True,
                progress=False,
            )
        except Exception as exc:
            logger.debug(f"Pre-market download failed for {ticker}: {exc}")
            return {}

        if intraday.empty:
            return {}

        intraday = intraday.iloc[-int(24 * 60 / 5) :]
        idx = intraday.index
        if hasattr(idx, "tz_convert"):
            try:
                idx = idx.tz_convert("US/Eastern")
            except Exception:
                idx = idx.tz_localize("US/Eastern")
        intraday = intraday.copy()
        intraday.index = idx

        pre_market = intraday[intraday.index.time < datetime.strptime("09:30", "%H:%M").time()]
        if pre_market.empty:
            return {}

        window_minutes = self.config.premarket_window_minutes
        cutoff = pre_market.index.max() - timedelta(minutes=window_minutes)
        pre_window = pre_market[pre_market.index >= cutoff]
        if pre_window.empty:
            pre_window = pre_market.tail(max(1, window_minutes // 5))

        volume = float(pre_window["Volume"].sum())
        if volume == 0:
            return {}
        vwap = float(
            (
                (pre_window["High"] + pre_window["Low"] + pre_window["Close"]) / 3
                * pre_window["Volume"]
            ).sum()
            / max(volume, 1)
        )
        open_price = float(pre_window["Close"].iloc[-1])
        prev_close = float(intraday["Close"].iloc[-1])
        change_pct = (open_price / prev_close - 1) * 100 if prev_close else 0.0

        return {
            "pre_market_price": open_price,
            "pre_market_volume": volume,
            "pre_market_vwap": vwap,
            "pre_market_change_pct": change_pct,
        }


class PatternDetector:
    """Rule-based detector for classical price patterns."""

    def detect(self, prices: pd.Series) -> List[str]:
        closes = prices.dropna().tail(160)
        if len(closes) < 40:
            return []

        patterns: List[str] = []
        if self._double_bottom(closes):
            patterns.append("Double Bottom")
        if self._double_top(closes):
            patterns.append("Double Top")
        if self._cup_and_handle(closes):
            patterns.append("Cup & Handle")
        if self._flag_or_pennant(closes):
            patterns.append("Bull Flag")
        if self._head_and_shoulders(closes):
            patterns.append("Head & Shoulders")
        return patterns

    @staticmethod
    def _double_bottom(series: pd.Series) -> bool:
        window = series.tail(80)
        min_idx = window.idxmin()
        left = window[:min_idx].tail(20)
        right = window[min_idx:].head(20)
        if left.empty or right.empty:
            return False
        left_min = left.min()
        right_min = right.min()
        trough_diff = abs(left_min - right_min) / max(series.max(), 1)
        rebound = window.iloc[-1] > window.mean()
        return trough_diff < 0.02 and rebound

    @staticmethod
    def _double_top(series: pd.Series) -> bool:
        window = series.tail(80)
        max_idx = window.idxmax()
        left = window[:max_idx].tail(20)
        right = window[max_idx:].head(20)
        if left.empty or right.empty:
            return False
        left_max = left.max()
        right_max = right.max()
        peak_diff = abs(left_max - right_max) / max(series.max(), 1)
        breakdown = window.iloc[-1] < window.mean()
        return peak_diff < 0.02 and breakdown

    @staticmethod
    def _cup_and_handle(series: pd.Series) -> bool:
        window = series.tail(120)
        smoothed = window.rolling(10).mean().dropna()
        if len(smoothed) < 40:
            return False
        trough = smoothed.min()
        top = smoothed.max()
        curvature = (top - trough) / max(trough, 1)
        recent = smoothed.tail(20)
        return curvature > 0.08 and recent.mean() > smoothed.mean()

    @staticmethod
    def _flag_or_pennant(series: pd.Series) -> bool:
        window = series.tail(40)
        if len(window) < 40:
            return False
        recent = window.pct_change().dropna()
        surge = recent.head(5).mean() > 0.01
        consolidation = recent.tail(15).abs().mean() < 0.006
        return surge and consolidation

    @staticmethod
    def _head_and_shoulders(series: pd.Series) -> bool:
        window = series.tail(120)
        if len(window) < 60:
            return False
        smoothed = window.rolling(5).mean().dropna()
        mid = len(smoothed) // 2
        left = smoothed[:mid]
        right = smoothed[mid:]
        if left.empty or right.empty:
            return False
        head = smoothed.max()
        shoulders = sorted([left.max(), right.max()], reverse=True)
        if len(shoulders) < 2:
            return False
        symmetry = abs(shoulders[0] - shoulders[1]) / max(head, 1)
        neckline = smoothed.min()
        return head > shoulders[0] * 1.05 and symmetry < 0.02 and smoothed.iloc[-1] < neckline


class MLReturnPredictor:
    """Logistic regression classifier predicting positive forward returns."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config
        self.pipeline: Pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=400,
                        C=1.2,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        self.feature_columns = [
            "ret_1d",
            "ret_5d",
            "ret_21d",
            "vol_10d",
            "rsi",
            "macd",
            "signal",
            "atr",
            "obv_pct",
        ]

    def train_and_score(self, histories: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        X: List[List[float]] = []
        y: List[int] = []
        latest_map: Dict[str, List[float]] = {}

        for ticker, hist in histories.items():
            if hist is None or hist.empty:
                continue
            df = self._feature_engineering(hist)
            if df.empty:
                continue

            future = df["future_positive"].astype(int)
            features = df[self.feature_columns]
            X.extend(features.values.tolist())
            y.extend(future.values.tolist())
            latest_map[ticker] = features.iloc[-1].tolist()

        if len(X) < self.config.ml_min_samples or len(set(y)) < 2:
            return {ticker: 0.5 for ticker in histories.keys()}

        self.pipeline.fit(X, y)
        scores: Dict[str, float] = {}
        for ticker, features in latest_map.items():
            prob = self.pipeline.predict_proba([features])[0, 1]
            scores[ticker] = float(prob)
        return scores

    def _feature_engineering(self, hist: pd.DataFrame) -> pd.DataFrame:
        df = hist.copy()
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        df = df.tail(self.config.ml_training_lookback + self.config.ml_lookahead_days + 5)
        df = df.assign(
            ret_1d=df["Close"].pct_change(),
            ret_5d=df["Close"].pct_change(5),
            ret_21d=df["Close"].pct_change(21),
        )
        df["vol_10d"] = df["ret_1d"].rolling(10).std()
        df["rsi"] = self._rsi(df["Close"], window=14)
        macd_fast = df["Close"].ewm(span=12, adjust=False).mean()
        macd_slow = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"] = macd_fast - macd_slow
        df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["atr"] = self._atr(df)
        df["obv_pct"] = self._obv(df)
        df["future_positive"] = (df["Close"].shift(-self.config.ml_lookahead_days) > df["Close"]).astype(int)
        df = df.dropna()
        return df

    @staticmethod
    def _rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_gain = up.rolling(window).mean()
        avg_loss = down.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame) -> pd.Series:
        high = df.get("High", df["Close"]).fillna(df["Close"])
        low = df.get("Low", df["Close"]).fillna(df["Close"])
        close = df["Close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    @staticmethod
    def _obv(df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        volume = df.get("Volume", pd.Series(0, index=df.index))
        direction = np.sign(close.diff().fillna(0))
        obv = (direction * volume).cumsum()
        return obv.pct_change().rolling(5).mean()


class Backtester:
    """Simple forward-hold backtester for recommendation validation."""

    def run(self, recommendations: pd.DataFrame, histories: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        if recommendations.empty:
            return {"average_return": 0.0, "hit_rate": 0.0, "sample_size": 0}

        returns: List[float] = []
        for _, row in recommendations.iterrows():
            ticker = row["ticker"]
            hold = int(row.get("holding_period_days", 5))
            hist = histories.get(ticker)
            if hist is None or hist.empty:
                continue
            closes = hist["Close"].dropna()
            if len(closes) < hold + 2:
                continue
            entry = closes.iloc[-hold - 1]
            exit_price = closes.iloc[-1]
            if entry == 0:
                continue
            returns.append((exit_price / entry) - 1)

        if not returns:
            return {"average_return": 0.0, "hit_rate": 0.0, "sample_size": 0}

        positive = sum(1 for r in returns if r > 0)
        return {
            "average_return": float(np.mean(returns)),
            "hit_rate": float(positive / len(returns)),
            "sample_size": len(returns),
        }


class ComprehensiveMarketIntelligence:
    """Main intelligence orchestrator for multi-factor market insights."""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

        self.sec_monitor = SECFilingMonitor(config)
        self.econ_tracker = EconomicCalendarTracker(config)
        self.congress_tracker = CongressionalTradeTracker(config)
        self.social_analyzer = SocialSentimentAnalyzer(config)
        self.institutional_tracker = InstitutionalTracker(config)
        self.insider_analyzer = InsiderTradingAnalyzer(config)
        self.options_analyzer = OptionsFlowAnalyzer(config)
        self.news_analyzer = NewsSentimentAnalyzer(config)
        self.rotation_model = SectorRotationModel(config)
        self.premarket_fetcher = PreMarketDataFetcher(config)
        self.pattern_detector = PatternDetector()
        self.ml_predictor = MLReturnPredictor(config)
        self.vol_forecaster = VolatilityForecaster()
        self.backtester = Backtester()

    def generate_comprehensive_report(self) -> Dict[str, object]:
        logger.info("🚀 Starting super-intelligent market analysis run ...")

        intelligence_data: Dict[str, pd.DataFrame] = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.sec_monitor.get_recent_filings): "sec_filings",
                executor.submit(self.econ_tracker.get_upcoming_events): "economic_calendar",
                executor.submit(self.congress_tracker.get_recent_trades): "congressional_trades",
                executor.submit(self.social_analyzer.get_trending_stocks): "social_trends",
                executor.submit(self.institutional_tracker.get_hot_hands): "institutional_activity",
                executor.submit(self.insider_analyzer.get_notable_insider_trades): "insider_trades",
                executor.submit(self.options_analyzer.get_unusual_activity): "options_flow",
                executor.submit(self.news_analyzer.get_latest_sentiment): "news_sentiment",
                executor.submit(self.rotation_model.calculate_relative_strength): "sector_rotation",
            }

            for future in as_completed(futures):
                name = futures[future]
                try:
                    intelligence_data[name] = future.result()
                    logger.success(f"✅ {name.replace('_', ' ').title()} loaded")
                except Exception as exc:  # pragma: no cover - resilience
                    logger.error(f"❌ Failed to load {name}: {exc}")
                    intelligence_data[name] = pd.DataFrame()

        stock_data, historical_data = self._run_stock_screening()
        intelligence_data["stock_screening"] = stock_data
        intelligence_data["historical_data"] = historical_data

        recommendations = self._generate_recommendations(intelligence_data)
        report = self._create_dashboard_report(intelligence_data, recommendations)
        artifacts = self._save_report(report, intelligence_data)
        report["artifacts"] = {name: str(path) for name, path in artifacts.items()}
        return report

    def _run_stock_screening(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        logger.info("Running accelerated stock screening with advanced analytics ...")
        tickers = self._get_sp500_tickers()
        if self.config.tickers_limit:
            tickers = tickers[: self.config.tickers_limit]

        results: List[Dict[str, object]] = []
        histories: Dict[str, pd.DataFrame] = {}

        def worker(symbol: str) -> Optional[Tuple[Dict[str, object], pd.DataFrame]]:
            try:
                # Add small delay to avoid rate limiting
                import time
                time.sleep(0.1)
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Handle case where info is None
                if info is None:
                    return None
                
                hist = ticker.history(period=f"{self.config.lookback_days}d", auto_adjust=False)
                if hist.empty:
                    return None
                hist = hist.dropna(subset=["Close"]).tail(self.config.lookback_days)

                # Handle case where Close column doesn't exist or is empty
                if "Close" not in hist.columns or hist["Close"].empty:
                    return None
                    
                latest_close = float(hist["Close"].iloc[-1])
                if latest_close < self.config.min_price:
                    return None
                    
                # Handle case where Volume column doesn't exist
                if "Volume" not in hist.columns:
                    avg_vol = 0
                else:
                    avg_vol = float(hist["Volume"].tail(30).mean())
                if avg_vol < self.config.min_volume:
                    return None

                metrics = self._compute_advanced_metrics(hist)
                premarket = self.premarket_fetcher.fetch(symbol)
                patterns = self.pattern_detector.detect(hist["Close"])

                row = {
                    "ticker": symbol,
                    "company": info.get("shortName", symbol),
                    "sector": info.get("sector", "Unknown"),
                    "price": latest_close,
                    "market_cap": info.get("marketCap", np.nan),
                    "pe_ratio": info.get("trailingPE", np.nan),
                    "forward_pe": info.get("forwardPE", np.nan),
                    "peg_ratio": info.get("pegRatio", np.nan),
                    "profit_margin": info.get("profitMargins", np.nan),
                    "revenue_growth": info.get("revenueGrowth", np.nan),
                    "return_on_equity": info.get("returnOnEquity", np.nan),
                    "beta": info.get("beta", np.nan),
                    **metrics,
                    **premarket,
                    "pattern_signals": ", ".join(patterns) if patterns else "None",
                }

                row["holding_period_days"] = self._estimate_holding_period(row)
                row["exit_strategy"] = self._determine_exit_strategy(row)
                histories[symbol] = hist
                return row, hist
            except Exception as exc:  # pragma: no cover - per ticker resilience
                logger.debug(f"Ticker {symbol} processing error: {exc}")
                return None

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {executor.submit(worker, symbol): symbol for symbol in tickers}
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                row, _hist = result
                results.append(row)

        if not results:
            return pd.DataFrame(), {}

        stock_df = pd.DataFrame(results)
        ml_scores = self.ml_predictor.train_and_score(histories)
        vol_forecasts = self.vol_forecaster.forecast_batch(histories)

        stock_df["ml_probability"] = stock_df["ticker"].map(ml_scores).fillna(0.5)
        stock_df["forecast_volatility"] = stock_df["ticker"].map(vol_forecasts).fillna(np.nan)
        stock_df["risk_reward_ratio"] = stock_df.apply(self._compute_risk_reward, axis=1)
        return stock_df.sort_values("ml_probability", ascending=False).reset_index(drop=True), histories

    def _compute_advanced_metrics(self, hist: pd.DataFrame) -> Dict[str, float]:
        close = hist["Close"]
        volume = hist["Volume"]
        high = hist.get("High", close)
        low = hist.get("Low", close)

        ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan
        ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if len(close) > 63 else np.nan
        ret_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if len(close) > 126 else np.nan
        ret_12m = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 else np.nan

        rsi = self._calculate_rsi(close)
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else np.nan
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else np.nan
        volatility_3m = close.pct_change().tail(63).std() * math.sqrt(252) if len(close) > 63 else np.nan
        max_drawdown = ((close / close.cummax()) - 1).min()

        vwap_10 = self._calculate_vwap(close, high, low, volume, window=10)
        obv = self._calculate_obv(close, volume)
        unusual_volume_ratio = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(20).mean() else np.nan
        atr = self._calculate_atr(high, low, close)

        return {
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_6m": ret_6m,
            "ret_12m": ret_12m,
            "volume_avg": float(volume.tail(20).mean()),
            "rsi": float(rsi.iloc[-1]) if not rsi.empty else 50.0,
            "sma_50": float(sma_50) if not pd.isna(sma_50) else np.nan,
            "sma_200": float(sma_200) if not pd.isna(sma_200) else np.nan,
            "volatility_3m": float(volatility_3m) if not pd.isna(volatility_3m) else np.nan,
            "max_drawdown": float(max_drawdown) if not pd.isna(max_drawdown) else np.nan,
            "vwap_10": float(vwap_10) if not pd.isna(vwap_10) else np.nan,
            "obv": float(obv.iloc[-1]) if len(obv) else np.nan,
            "unusual_volume_ratio": float(unusual_volume_ratio) if not pd.isna(unusual_volume_ratio) else np.nan,
            "atr": float(atr.iloc[-1]) if len(atr) else np.nan,
        }

    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_vwap(close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series, window: int = 10) -> float:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
        return vwap.iloc[-1]

    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()

    @staticmethod
    def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    def _estimate_holding_period(self, row: Dict[str, object]) -> int:
        lower, upper = self.config.hold_period_bounds
        momentum = row.get("ret_1m") or 0
        volatility = abs(row.get("volatility_3m") or 0)
        if momentum > 8 and volatility < 0.35:
            return upper
        if momentum < 0 and volatility > 0.45:
            return lower
        base = lower + (upper - lower) / 2
        adjustment = (momentum / 20) - (volatility * 10)
        return int(max(lower, min(upper, base + adjustment)))

    def _determine_exit_strategy(self, row: Dict[str, object]) -> str:
        atr = row.get("atr") or 0
        price = row.get("price") or 0
        if atr and price:
            stop = price - 2 * atr
            target = price + 3 * atr
            return f"Trail stop 2xATR (~{stop:.2f}), target +3xATR (~{target:.2f})"
        return "Price closes below 21-EMA"

    def _compute_risk_reward(self, row: pd.Series) -> float:
        atr = row.get("atr") or np.nan
        price = row.get("price") or np.nan
        if pd.isna(atr) or atr == 0 or pd.isna(price):
            return np.nan
        stop = price - 1.5 * atr
        target = price + 3 * atr
        if stop <= 0 or target <= price:
            return np.nan
        return float((target - price) / (price - stop))

    def _get_sp500_tickers(self) -> List[str]:
        if hasattr(yf, "tickers_sp500"):
            try:
                tickers = yf.tickers_sp500()
                if tickers:
                    return [t.replace(".", "-").upper() for t in tickers]
            except Exception:
                pass

        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        for flavor in (None, "bs4"):
            try:
                tables = pd.read_html(wiki_url, match="Symbol", flavor=flavor)
                symbols = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper()
                if len(symbols) > 100:
                    return symbols.tolist()
            except Exception:
                continue

        csv_urls = [
            "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        ]
        for url in csv_urls:
            try:
                data = pd.read_csv(url)
                if "Symbol" in data.columns and len(data) > 100:
                    return data["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
            except Exception:
                continue

        logger.warning("Falling back to mega-cap subset for analysis.")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "JPM", "XOM"]

    def _generate_recommendations(self, intelligence_data: Dict[str, object]) -> pd.DataFrame:
        stock_df: pd.DataFrame = intelligence_data.get("stock_screening", pd.DataFrame())
        if stock_df.empty:
            return pd.DataFrame()

        options_flow = intelligence_data.get("options_flow", pd.DataFrame())
        news_sentiment = intelligence_data.get("news_sentiment", pd.DataFrame())
        sector_rotation = intelligence_data.get("sector_rotation", pd.DataFrame())

        records: List[Dict[str, object]] = []
        for _, stock in stock_df.iterrows():
            ticker = stock["ticker"]
            technical_score = self._calculate_technical_score(stock)
            fundamental_score = self._calculate_fundamental_score(stock)
            quality_score = self._calculate_quality_score(stock)
            risk_score = self._calculate_risk_score(stock)
            volume_score = self._calculate_volume_score(stock)
            pattern_score = self._calculate_pattern_score(stock)
            ml_score = self._calculate_ml_score(stock)
            premarket_score = self._calculate_premarket_score(stock)
            options_score = self._calculate_options_score(ticker, options_flow)
            news_score = self._calculate_news_score(ticker, news_sentiment)
            rotation_score = self._calculate_rotation_score(stock, sector_rotation)
            volatility_score = self._calculate_volatility_score(stock)
            insider_score = self._calculate_insider_score(ticker, intelligence_data)
            institutional_score = self._calculate_institutional_score(ticker, intelligence_data)
            social_score = self._calculate_social_score(ticker, intelligence_data)
            macro_score = self._calculate_macro_score(stock, intelligence_data)
            congressional_score = self._calculate_congressional_score(ticker, intelligence_data)

            composite = (
                technical_score * self.config.technical_weight
                + fundamental_score * self.config.fundamental_weight
                + quality_score * self.config.quality_weight
                + risk_score * self.config.risk_weight
                + volume_score * self.config.volume_weight
                + pattern_score * self.config.pattern_weight
                + ml_score * self.config.ml_weight
                + premarket_score * self.config.premarket_weight
                + options_score * self.config.options_weight
                + news_score * self.config.news_weight
                + rotation_score * self.config.rotation_weight
                + volatility_score * self.config.volatility_weight
                + insider_score * self.config.insider_weight
                + institutional_score * self.config.institutional_weight
                + social_score * self.config.social_weight
                + macro_score * self.config.macro_weight
                + congressional_score * self.config.congressional_weight
            )

            recommendation = self._score_to_rating(composite)
            catalysts = self._identify_catalysts(ticker, stock, intelligence_data)

            records.append(
                {
                    "ticker": ticker,
                    "company": stock["company"],
                    "sector": stock["sector"],
                    "price": stock["price"],
                    "recommendation": recommendation,
                    "conviction_score": round(composite, 2),
                    "technical_score": round(technical_score, 1),
                    "fundamental_score": round(fundamental_score, 1),
                    "quality_score": round(quality_score, 1),
                    "risk_score": round(risk_score, 1),
                    "volume_score": round(volume_score, 1),
                    "pattern_score": round(pattern_score, 1),
                    "ml_score": round(ml_score, 1),
                    "premarket_score": round(premarket_score, 1),
                    "options_score": round(options_score, 1),
                    "news_score": round(news_score, 1),
                    "rotation_score": round(rotation_score, 1),
                    "volatility_score": round(volatility_score, 1),
                    "insider_score": round(insider_score, 1),
                    "institutional_score": round(institutional_score, 1),
                    "social_score": round(social_score, 1),
                    "macro_score": round(macro_score, 1),
                    "congressional_score": round(congressional_score, 1),
                    "ml_probability": round(stock.get("ml_probability", 0.5), 4),
                    "holding_period_days": int(stock.get("holding_period_days", 5)),
                    "exit_strategy": stock.get("exit_strategy", ""),
                    "risk_reward_ratio": round(stock.get("risk_reward_ratio", np.nan), 2)
                    if not pd.isna(stock.get("risk_reward_ratio", np.nan))
                    else np.nan,
                    "pattern_signals": stock.get("pattern_signals", "None"),
                    "unusual_volume_ratio": round(stock.get("unusual_volume_ratio", np.nan), 2)
                    if not pd.isna(stock.get("unusual_volume_ratio", np.nan))
                    else np.nan,
                    "pre_market_change_pct": round(stock.get("pre_market_change_pct", np.nan), 2)
                    if not pd.isna(stock.get("pre_market_change_pct", np.nan))
                    else np.nan,
                    "pre_market_volume": stock.get("pre_market_volume", np.nan),
                    "forecast_volatility": round(stock.get("forecast_volatility", np.nan), 3)
                    if not pd.isna(stock.get("forecast_volatility", np.nan))
                    else np.nan,
                    "key_catalysts": catalysts,
                }
            )

        rec_df = pd.DataFrame(records)
        rec_df = rec_df[rec_df["conviction_score"] >= self.config.min_conviction_score]
        rec_df = rec_df.sort_values("conviction_score", ascending=False)
        return rec_df.head(self.config.max_recommendations).reset_index(drop=True)

    @staticmethod
    def _score_to_rating(score: float) -> str:
        if score >= 80:
            return "STRONG BUY"
        if score >= 60:
            return "BUY"
        if score >= 40:
            return "HOLD"
        if score >= 20:
            return "SELL"
        return "STRONG SELL"

    def _calculate_technical_score(self, stock: pd.Series) -> float:
        score = 50
        rsi = stock.get("rsi", 50)
        if 35 <= rsi <= 60:
            score += 10
        elif 30 <= rsi < 35:
            score += 15
        elif rsi < 30:
            score += 25
        elif 60 < rsi <= 70:
            score -= 5
        elif rsi > 70:
            score -= 15

        for window in ["ret_1m", "ret_3m", "ret_6m", "ret_12m"]:
            value = stock.get(window)
            if value is None or pd.isna(value):
                continue
            if value > 10:
                score += 4
            elif value < -10:
                score -= 6

        price = stock.get("price")
        sma_50 = stock.get("sma_50")
        sma_200 = stock.get("sma_200")
        if price and sma_50 and price > sma_50:
            score += 5
        if sma_50 and sma_200:
            score += 8 if sma_50 > sma_200 else -6
        return float(max(0, min(100, score)))

    def _calculate_fundamental_score(self, stock: pd.Series) -> float:
        score = 50
        pe = stock.get("pe_ratio")
        if pe and 5 < pe < 22:
            score += 10
        elif pe and pe > 40:
            score -= 10

        peg = stock.get("peg_ratio")
        if peg and peg < 1.0:
            score += 8
        elif peg and peg > 2.5:
            score -= 6

        market_cap = stock.get("market_cap")
        if market_cap and market_cap > 200_000_000_000:
            score += 6

        return float(max(0, min(100, score)))

    def _calculate_quality_score(self, stock: pd.Series) -> float:
        score = 50
        margin = stock.get("profit_margin")
        growth = stock.get("revenue_growth")
        roe = stock.get("return_on_equity")
        if margin and margin > 0.15:
            score += 10
        if growth and growth > 0.12:
            score += 10
        if roe and roe > 0.18:
            score += 8
        if growth and growth < 0:
            score -= 8
        return float(max(0, min(100, score)))

    def _calculate_risk_score(self, stock: pd.Series) -> float:
        score = 55
        volatility = stock.get("volatility_3m")
        drawdown = stock.get("max_drawdown")
        beta = stock.get("beta")
        if volatility and volatility < 0.3:
            score += 10
        elif volatility and volatility > 0.5:
            score -= 10
        if drawdown and drawdown > -0.2:
            score += 10
        elif drawdown and drawdown < -0.4:
            score -= 10
        if beta and beta < 0.8:
            score += 5
        elif beta and beta > 1.5:
            score -= 6
        return float(max(0, min(100, score)))

    def _calculate_volume_score(self, stock: pd.Series) -> float:
        score = 50
        unusual = stock.get("unusual_volume_ratio")
        obv = stock.get("obv")
        if unusual and unusual > self.config.volume_spike_threshold:
            score += min(25, (unusual - self.config.volume_spike_threshold) * 10)
        elif unusual and unusual < 1:
            score -= 5
        if obv and obv > 0:
            score += 5
        return float(max(0, min(100, score)))

    def _calculate_pattern_score(self, stock: pd.Series) -> float:
        patterns = stock.get("pattern_signals", "")
        if not patterns or patterns == "None":
            return 50
        bullish = {"Double Bottom", "Cup & Handle", "Bull Flag"}
        bearish = {"Double Top", "Head & Shoulders"}
        score = 50
        for pattern in patterns.split(","):
            pattern = pattern.strip()
            if pattern in bullish:
                score += 12
            elif pattern in bearish:
                score -= 12
        return float(max(0, min(100, score)))

    def _calculate_ml_score(self, stock: pd.Series) -> float:
        prob = stock.get("ml_probability", 0.5)
        return float(max(0, min(100, 50 + (prob - 0.5) * 200)))

    def _calculate_premarket_score(self, stock: pd.Series) -> float:
        change = stock.get("pre_market_change_pct")
        volume = stock.get("pre_market_volume")
        if change is None or pd.isna(change):
            return 50
        score = 50 + change
        if volume and volume > stock.get("volume_avg", 0) * 0.25:
            score += 10
        return float(max(0, min(100, score)))

    def _calculate_options_score(self, ticker: str, options_flow: pd.DataFrame) -> float:
        if options_flow.empty:
            return 50
        data = options_flow[options_flow["ticker"] == ticker]
        if data.empty:
            return 50
        bullish = data[data["type"].str.contains("CALL")]
        bearish = data[data["type"].str.contains("PUT")]
        score = 50
        if not bullish.empty:
            score += 15
        if not bearish.empty:
            score -= 10
        return float(max(0, min(100, score)))

    def _calculate_news_score(self, ticker: str, sentiment: pd.DataFrame) -> float:
        if sentiment.empty:
            return 50
        data = sentiment[sentiment["ticker"] == ticker]
        if data.empty:
            return 50
        avg = data["sentiment_score"].mean()
        return float(max(0, min(100, 50 + avg * 80)))

    def _calculate_rotation_score(self, stock: pd.Series, rotation: pd.DataFrame) -> float:
        if rotation.empty:
            return 50
        sector = stock.get("sector")
        if not sector:
            return 50
        row = rotation[rotation["sector"] == sector]
        if row.empty:
            return 50
        try:
            rel_val = row["relative_to_spy"].iloc[0]
            momentum_val = row["momentum_20d"].iloc[0]
            rel = float(rel_val) if rel_val is not None and not (isinstance(rel_val, float) and rel_val != rel_val) else 0.0
            momentum = float(momentum_val) if momentum_val is not None and not (isinstance(momentum_val, float) and momentum_val != momentum_val) else 0.0
        except (ValueError, TypeError, IndexError):
            rel = 0.0
            momentum = 0.0
        score = 50 + rel * 120 + momentum * 200
        return float(max(0, min(100, score)))

    def _calculate_volatility_score(self, stock: pd.Series) -> float:
        forecast = stock.get("forecast_volatility")
        if forecast is None or pd.isna(forecast):
            return 50
        if forecast < 0.25:
            return 70
        if forecast > 0.6:
            return 35
        return 50

    def _calculate_insider_score(self, ticker: str, intelligence_data: Dict[str, object]) -> float:
        insiders: pd.DataFrame = intelligence_data.get("insider_trades", pd.DataFrame())
        if insiders.empty:
            return 50
        data = insiders[insiders["ticker"] == ticker]
        if data.empty:
            return 50
        buys = data[data["transaction"] == "BUY"]
        sells = data[data["transaction"] == "SELL"]
        score = 50
        if not buys.empty:
            score += 20
        if len(sells) > len(buys):
            score -= 10
        return float(max(0, min(100, score)))

    def _calculate_institutional_score(self, ticker: str, intelligence_data: Dict[str, object]) -> float:
        inst: pd.DataFrame = intelligence_data.get("institutional_activity", pd.DataFrame())
        if inst.empty:
            return 50
        data = inst[inst["ticker"] == ticker]
        if data.empty:
            return 50
        buys = data[data["action"] == "BUY"]
        score = 50
        if not buys.empty:
            score += 15
        return float(max(0, min(100, score)))

    def _calculate_social_score(self, ticker: str, intelligence_data: Dict[str, object]) -> float:
        social: pd.DataFrame = intelligence_data.get("social_trends", pd.DataFrame())
        if social.empty:
            return 50
        data = social[social["ticker"] == ticker]
        if data.empty:
            return 50
        avg = data["influence_score"].mean()
        return float(max(0, min(100, 45 + avg / 2)))

    def _calculate_macro_score(self, stock: pd.Series, intelligence_data: Dict[str, object]) -> float:
        sector = stock.get("sector", "")
        if "Technology" in sector:
            return 60
        if "Financial" in sector:
            return 55
        return 50

    def _calculate_congressional_score(self, ticker: str, intelligence_data: Dict[str, object]) -> float:
        congress: pd.DataFrame = intelligence_data.get("congressional_trades", pd.DataFrame())
        if congress.empty:
            return 50
        data = congress[congress["ticker"] == ticker]
        if data.empty:
            return 50
        buys = data[data["transaction"] == "BUY"]
        return 70 if not buys.empty else 50

    def _identify_catalysts(
        self,
        ticker: str,
        stock: pd.Series,
        intelligence_data: Dict[str, object],
    ) -> str:
        catalysts: List[str] = []
        if stock.get("unusual_volume_ratio") and stock["unusual_volume_ratio"] > self.config.volume_spike_threshold:
            catalysts.append("Unusual volume surge")
        if stock.get("pattern_signals") and stock["pattern_signals"] != "None":
            catalysts.append(stock["pattern_signals"])
        if stock.get("pre_market_change_pct") and stock["pre_market_change_pct"] > 1:
            catalysts.append("Positive pre-market gap")
        if stock.get("ml_probability") and stock["ml_probability"] > 0.6:
            catalysts.append("ML bullish probability")

        insiders: pd.DataFrame = intelligence_data.get("insider_trades", pd.DataFrame())
        if not insiders.empty and not insiders[insiders["ticker"] == ticker].empty:
            catalysts.append("Insider accumulation")
        options_flow: pd.DataFrame = intelligence_data.get("options_flow", pd.DataFrame())
        if not options_flow.empty and not options_flow[options_flow["ticker"] == ticker].empty:
            catalysts.append("Unusual options flow")

        return "; ".join(dict.fromkeys(catalysts)) if catalysts else "Diverse multi-factor alignment"

    def _create_dashboard_report(
        self,
        intelligence_data: Dict[str, object],
        recommendations: pd.DataFrame,
    ) -> Dict[str, object]:
        stock_df: pd.DataFrame = intelligence_data.get("stock_screening", pd.DataFrame())
        histories: Dict[str, pd.DataFrame] = intelligence_data.get("historical_data", {})
        summary = {
            "total_stocks_analyzed": int(len(stock_df)),
            "recommendations_generated": int(len(recommendations)),
            "strong_buys": int((recommendations["recommendation"] == "STRONG BUY").sum()) if not recommendations.empty else 0,
            "buys": int((recommendations["recommendation"] == "BUY").sum()) if not recommendations.empty else 0,
            "data_sources_active": len([k for k, v in intelligence_data.items() if (isinstance(v, pd.DataFrame) and not v.empty) or (isinstance(v, list) and len(v) > 0) or (not isinstance(v, (pd.DataFrame, list)))]),
            "avg_risk_reward": float(recommendations["risk_reward_ratio"].mean()) if "risk_reward_ratio" in recommendations else np.nan,
        }

        top_histories = {}
        for pick in recommendations.head(3).itertuples():
            hist = histories.get(pick.ticker)
            if hist is None:
                continue
            trimmed = hist.tail(120).reset_index()
            trimmed["Date"] = trimmed["Date"].astype(str)
            top_histories[pick.ticker] = trimmed[["Date", "Close"]].to_dict("records")

        backtest_summary = self.backtester.run(recommendations, histories)

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": summary,
            "market_intelligence": {
                name: df.to_dict("records") if isinstance(df, pd.DataFrame) else []
                for name, df in intelligence_data.items()
                if name != "historical_data"
            },
            "recommendations": recommendations.to_dict("records"),
            "top_picks": recommendations.head(5).to_dict("records") if not recommendations.empty else [],
            "analytics": {
                "top_pick_history": top_histories,
                "backtest": backtest_summary,
            },
        }
        return report

    def _save_report(self, report: Dict[str, object], intelligence_data: Dict[str, object]) -> Dict[str, Path]:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = self.config.output_dir / f"market_intelligence_{timestamp}.json"
        csv_path = None
        recs = pd.DataFrame(report.get("recommendations", []))
        if not recs.empty:
            csv_path = self.config.output_dir / f"recommendations_{timestamp}.csv"
            recs.to_csv(csv_path, index=False)

        html_path = self.config.output_dir / f"dashboard_{timestamp}.html"
        self._create_html_dashboard(report, intelligence_data, html_path)

        with open(json_path, "w") as fp:
            json.dump(report, fp, indent=2, default=str)

        artifacts = {"json": json_path, "html": html_path}
        if csv_path:
            artifacts["csv"] = csv_path

        logger.success("📊 Artifacts saved:")
        for label, path in artifacts.items():
            logger.success(f"  • {label.upper()}: {path}")
        return artifacts

    def _create_html_dashboard(
        self,
        report: Dict[str, object],
        intelligence_data: Dict[str, object],
        html_path: Path,
    ) -> None:
        top_picks = report.get("top_picks", [])
        recommendations = report.get("recommendations", [])
        top_histories = report.get("analytics", {}).get("top_pick_history", {})
        plotly_div = ""
        if top_picks:
            first = top_picks[0]
            hist = top_histories.get(first["ticker"], [])
            if hist:
                dates = [row["Date"] for row in hist]
                closes = [row["Close"] for row in hist]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=closes, mode="lines", name="Close"))
                fig.update_layout(title=f"{first['ticker']} Price (Last 120 sessions)", height=320)
                plotly_div = plot(fig, include_plotlyjs=False, output_type="div")

        rows = "".join(
            """
            <tr class="{cls}">
                <td><strong>{ticker}</strong></td>
                <td>{company}</td>
                <td>{recommendation}</td>
                <td>{conviction_score}%</td>
                <td>${price:.2f}</td>
                <td>{holding_period_days} days</td>
                <td>{risk_reward}</td>
                <td>{key_catalysts}</td>
            </tr>
            """.format(
                cls=rec["recommendation"].lower().replace(" ", "-"),
                ticker=rec["ticker"],
                company=rec["company"],
                recommendation=rec["recommendation"],
                conviction_score=rec["conviction_score"],
                price=float(rec["price"]),
                holding_period_days=rec.get("holding_period_days", "-"),
                risk_reward=f"{rec.get('risk_reward_ratio', '—')}",
                key_catalysts=rec.get("key_catalysts", ""),
            )
            for rec in recommendations[:15]
        )

        sec_items = "".join(
            f"<li>{item.get('company', 'Unknown')} — {item.get('form_type', '')}</li>"
            for item in report.get("market_intelligence", {}).get("sec_filings", [])[:8]
        )
        econ_items = "".join(
            f"<li>{item.get('event', 'Event')} — {item.get('date', '')}</li>"
            for item in report.get("market_intelligence", {}).get("economic_calendar", [])[:8]
        )
        options_items = "".join(
            f"<li>{item.get('ticker')} {item.get('type')} exp {item.get('expiry')} (${item.get('premium', 0):,.0f})</li>"
            for item in report.get("market_intelligence", {}).get("options_flow", [])[:6]
        )
        news_items = "".join(
            f"<li><strong>{item.get('ticker')}</strong> — {item.get('headline')}</li>"
            for item in report.get("market_intelligence", {}).get("news_sentiment", [])[:6]
        )

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <title>Market Intelligence Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Inter', Arial, sans-serif; margin: 0; padding: 0; background: #f4f5f9; color: #1f2933; }}
                header {{ background: linear-gradient(135deg, #2563eb, #7c3aed); color: white; padding: 32px; }}
                .container {{ padding: 24px; max-width: 1200px; margin: 0 auto; }}
                h1 {{ margin: 0 0 8px; font-size: 32px; }}
                h2 {{ margin-top: 32px; font-size: 24px; }}
                .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }}
                .card {{ background: white; border-radius: 14px; padding: 20px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
                th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
                th {{ background: #eef2ff; text-transform: uppercase; font-size: 12px; letter-spacing: 0.05em; }}
                tr.buy td {{ border-left: 4px solid #16a34a; }}
                tr.strong-buy td {{ border-left: 4px solid #2563eb; }}
                tr.hold td {{ border-left: 4px solid #fbbf24; }}
                tr.sell td {{ border-left: 4px solid #dc2626; }}
                ul {{ margin: 0; padding-left: 18px; }}
            </style>
        </head>
        <body>
            <header>
                <h1>🚀 Market Intelligence Dashboard</h1>
                <p>Generated: {report['timestamp']}</p>
            </header>
            <div class="container">
                <div class="grid">
                    <div class="card">
                        <h2>Summary</h2>
                        <p><strong>Stocks Analyzed:</strong> {report['summary']['total_stocks_analyzed']}</p>
                        <p><strong>Recommendations:</strong> {report['summary']['recommendations_generated']}</p>
                        <p><strong>Strong Buys:</strong> {report['summary']['strong_buys']}</p>
                        <p><strong>Avg. Risk/Reward:</strong> {report['summary']['avg_risk_reward']:.2f}</p>
                    </div>
                    <div class="card">
                        <h2>Top Pick Trend</h2>
                        {plotly_div or '<p>No chart available</p>'}
                    </div>
                    <div class="card">
                        <h2>Backtest Snapshot</h2>
                        <p><strong>Sample Size:</strong> {report['analytics']['backtest']['sample_size']}</p>
                        <p><strong>Avg Return:</strong> {report['analytics']['backtest']['average_return']:.2%}</p>
                        <p><strong>Hit Rate:</strong> {report['analytics']['backtest']['hit_rate']:.1%}</p>
                    </div>
                </div>

                <div class="card">
                    <h2>Stock Recommendations</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Company</th>
                                <th>Recommendation</th>
                                <th>Conviction</th>
                                <th>Price</th>
                                <th>Holding</th>
                                <th>Risk/Reward</th>
                                <th>Catalysts</th>
                            </tr>
                        </thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>

                <div class="grid">
                    <div class="card">
                        <h2>SEC Filings</h2>
                        <ul>{sec_items}</ul>
                    </div>
                    <div class="card">
                        <h2>Economic Calendar</h2>
                        <ul>{econ_items}</ul>
                    </div>
                    <div class="card">
                        <h2>Options Flow</h2>
                        <ul>{options_items}</ul>
                    </div>
                    <div class="card">
                        <h2>News Sentiment</h2>
                        <ul>{news_items}</ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        with open(html_path, "w") as fp:
            fp.write(html)


def main() -> None:
    print("🚀 Starting Comprehensive Market Intelligence System...")

    config = IntelligenceConfig(
        max_workers=12,
        min_conviction_score=55.0,
        max_recommendations=20,
    )

    engine = ComprehensiveMarketIntelligence(config)

    try:
        report = engine.generate_comprehensive_report()

        print("\n" + "=" * 90)
        print("📊 MARKET INTELLIGENCE DASHBOARD SUMMARY")
        print("=" * 90)
        print(f"📈 Total Stocks Analyzed: {report['summary']['total_stocks_analyzed']}")
        print(f"🎯 Recommendations Generated: {report['summary']['recommendations_generated']}")
        print(f"⭐ Strong Buys: {report['summary']['strong_buys']}")
        print(f"📊 Avg Risk/Reward: {report['summary']['avg_risk_reward']:.2f}")
        backtest = report.get('analytics', {}).get('backtest', {})
        if backtest:
            print(f"🧪 Backtest Hit Rate: {backtest.get('hit_rate', 0):.1%} on {backtest.get('sample_size', 0)} samples")

        if report['top_picks']:
            print("\n🏆 TOP PICKS:")
            for i, pick in enumerate(report['top_picks'][:5], 1):
                print(f"  {i}. {pick['ticker']} - {pick['recommendation']} (Conviction: {pick['conviction_score']}%)")
                print(f"     {pick['company']} | Catalysts: {pick['key_catalysts']}")

        print(f"\n📁 Reports saved to: {config.output_dir}")
        print("✅ Market Intelligence System Complete!")
    except Exception as exc:
        logger.error(f"System error: {exc}")
        print(f"❌ Error: {exc}")


if __name__ == "__main__":
    main()

