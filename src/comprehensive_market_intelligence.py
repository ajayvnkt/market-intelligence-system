
# ===================== Comprehensive Market Intelligence Dashboard =====================
# 
# Integrated system combining:
# - Stock screening with technical/fundamental analysis
# - SEC filing monitoring (8-K, 10-K, 10-Q, Form 4)
# - Economic calendar tracking (US + Global)
# - Congressional trading activity
# - Social media sentiment analysis
# - Institutional investor tracking (13F filings)
# - Insider trading analysis
# 
# All integrated into a single recommendation engine for buy/sell decisions
# 
# Dependencies: pip install yfinance pandas numpy requests beautifulsoup4 
#              feedparser tweepy pynance sec-edgar-downloader fredapi
# ===============================================================================

from __future__ import annotations
import asyncio
import json
import re
import warnings
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import gc

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger
import feedparser
from urllib.parse import urljoin
 
# Optional caching (speeds up repeated runs)
try:
    import requests_cache  # type: ignore
    requests_cache.install_cache("yf_cache", expire_after=3600)
except Exception:
    pass

# Suppress warnings
warnings.filterwarnings('ignore')
pd.options.mode.copy_on_write = True

# ========================= Enhanced Configuration =========================
@dataclass
class IntelligenceConfig:
    """Comprehensive configuration for market intelligence system"""

    # Stock screening config
    lookback_days: int = 252
    chunk_size: int = 50
    max_workers: int = 8
    min_price: float = 10.0
    min_volume: int = 500000

    # Limit number of tickers for faster test runs (None = no limit)
    tickers_limit: Optional[int] = None

    # Data source settings
    sec_edgar_delay_hours: int = 24
    economic_calendar_days: int = 7
    social_sentiment_hours: int = 48
    insider_trade_days: int = 2
    congressional_trade_days: int = 7
    institutional_filing_days: int = 30

    # Recommendation weights
    technical_weight: float = 0.25
    fundamental_weight: float = 0.20
    insider_weight: float = 0.15
    institutional_weight: float = 0.15
    social_weight: float = 0.10
    macro_weight: float = 0.10
    congressional_weight: float = 0.05

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path.cwd() / "market_intelligence")
    max_recommendations: int = 20
    min_conviction_score: float = 60.0

    # API keys (set these in environment or config file)
    sec_api_key: Optional[str] = None
    fred_api_key: Optional[str] = None
    twitter_bearer_token: Optional[str] = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.add(self.output_dir / "intelligence.log", rotation="10 MB", retention="30 days")
        # Allow overriding tickers_limit via env (e.g., CMI_TICKERS_LIMIT=3)
        if self.tickers_limit is None:
            env_limit = os.getenv("CMI_TICKERS_LIMIT")
            if env_limit:
                try:
                    self.tickers_limit = int(env_limit)
                except ValueError:
                    pass

# ========================= Data Source Integrations =========================

class SECFilingMonitor:
    """Monitor SEC filings for material disclosures"""

    def __init__(self, config: IntelligenceConfig):
        self.config = config
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar"

    def get_recent_filings(self) -> pd.DataFrame:
        """Get recent SEC filings from past 24 hours"""
        logger.info("Fetching recent SEC filings...")

        filings = []
        filing_types = ['8-K', '10-K', '10-Q', '4', '13D', '13G']

        try:
            for filing_type in filing_types:
                url = f"{self.base_url}?action=getcurrent&type={filing_type}"
                headers = {'User-Agent': 'Market Intelligence Bot 1.0'}

                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Parse filing table (simplified)
                    for row in soup.find_all('tr')[1:21]:  # Top 20 recent
                        cells = row.find_all('td')
                        if len(cells) >= 4:
                            filing_date = cells[0].text.strip()
                            company_name = cells[1].text.strip()
                            form_type = cells[2].text.strip()

                            # Extract ticker if available
                            ticker = self._extract_ticker(company_name)

                            filings.append({
                                'date': filing_date,
                                'ticker': ticker,
                                'company': company_name,
                                'form_type': form_type,
                                'potential_impact': self._assess_impact(form_type)
                            })

        except Exception as e:
            logger.error(f"SEC filing fetch error: {e}")

        return pd.DataFrame(filings)

    def _extract_ticker(self, company_name: str) -> Optional[str]:
        """Extract ticker from company name (simplified)"""
        # This would need enhancement with a proper company->ticker mapping
        return None

    def _assess_impact(self, form_type: str) -> str:
        """Assess potential market impact of filing type"""
        impact_map = {
            '8-K': 'HIGH',      # Material events
            '10-K': 'MEDIUM',   # Annual report
            '10-Q': 'MEDIUM',   # Quarterly report
            '4': 'MEDIUM',      # Insider trading
            '13D': 'HIGH',      # 5%+ ownership
            '13G': 'MEDIUM'     # Passive ownership
        }
        return impact_map.get(form_type, 'LOW')

class EconomicCalendarTracker:
    """Track economic calendar events"""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_upcoming_events(self) -> pd.DataFrame:
        """Get upcoming economic events for next 7 days"""
        logger.info("Fetching economic calendar...")

        # This would integrate with services like:
        # - FRED API for official data
        # - Economic calendar APIs
        # - Central bank websites

        events = []
        try:
            # Example events (would be fetched from actual sources)
            sample_events = [
                {
                    'date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                    'time': '08:30',
                    'event': 'CPI MoM',
                    'country': 'US',
                    'importance': 'HIGH',
                    'forecast': '0.2%',
                    'previous': '0.3%',
                    'impact': 'Inflation data affects Fed policy'
                },
                {
                    'date': (datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d'),
                    'time': '14:00',
                    'event': 'FOMC Rate Decision',
                    'country': 'US',
                    'importance': 'HIGH',
                    'forecast': '5.25%',
                    'previous': '5.25%',
                    'impact': 'Direct market impact on rates and equities'
                }
            ]
            events.extend(sample_events)

        except Exception as e:
            logger.error(f"Economic calendar fetch error: {e}")

        return pd.DataFrame(events)

class CongressionalTradeTracker:
    """Track congressional stock trades"""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_recent_trades(self) -> pd.DataFrame:
        """Get recent congressional trades"""
        logger.info("Fetching congressional trades...")

        trades = []
        try:
            # This would integrate with:
            # - House/Senate disclosure websites
            # - Third-party APIs like QuiverQuant
            # - Capitol Trades data

            sample_trades = [
                {
                    'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'member': 'Nancy Pelosi',
                    'party': 'D-CA',
                    'ticker': 'NVDA',
                    'transaction': 'BUY',
                    'amount': '$100K-250K',
                    'committee': 'Ways and Means',
                    'filing_delay': 15
                }
            ]
            trades.extend(sample_trades)

        except Exception as e:
            logger.error(f"Congressional trades fetch error: {e}")

        return pd.DataFrame(trades)

class SocialSentimentAnalyzer:
    """Analyze social media sentiment for viral trends"""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_trending_stocks(self) -> pd.DataFrame:
        """Get stocks trending on social media"""
        logger.info("Analyzing social sentiment...")

        trending = []
        try:
            # This would integrate with:
            # - Twitter API for mentions
            # - Reddit API for subreddit activity
            # - TikTok/Instagram APIs where available
            # - Google Trends API

            sample_trends = [
                {
                    'ticker': 'AAPL',
                    'platform': 'TikTok',
                    'trend_type': 'iPhone viral video',
                    'sentiment': 'POSITIVE',
                    'volume_spike': 150,  # % increase
                    'timeframe': '24h',
                    'influence_score': 85
                }
            ]
            trending.extend(sample_trends)

        except Exception as e:
            logger.error(f"Social sentiment fetch error: {e}")

        return pd.DataFrame(trending)

class InstitutionalTracker:
    """Track institutional investor activity"""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_hot_hands(self) -> pd.DataFrame:
        """Get notable institutional activity"""
        logger.info("Tracking institutional activity...")

        activity = []
        try:
            # This would integrate with:
            # - SEC 13F filings
            # - WhaleWisdom API
            # - Institutional holder databases

            sample_activity = [
                {
                    'date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                    'institution': 'ARK Invest',
                    'fund': 'ARKK',
                    'ticker': 'TSLA',
                    'action': 'BUY',
                    'shares': 500000,
                    'value': 85000000,
                    'conviction_level': 'HIGH'
                }
            ]
            activity.extend(sample_activity)

        except Exception as e:
            logger.error(f"Institutional tracking error: {e}")

        return pd.DataFrame(activity)

class InsiderTradingAnalyzer:
    """Analyze insider trading patterns"""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

    def get_notable_insider_trades(self) -> pd.DataFrame:
        """Get notable insider trades from Form 4 filings"""
        logger.info("Analyzing insider trades...")

        trades = []
        try:
            # This would parse actual Form 4 filings
            sample_trades = [
                {
                    'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    'ticker': 'AMZN',
                    'insider_name': 'Andy Jassy',
                    'title': 'CEO',
                    'transaction': 'BUY',
                    'shares': 10000,
                    'price': 145.50,
                    'value': 1455000,
                    'signal_strength': 'STRONG_BUY'
                }
            ]
            trades.extend(sample_trades)

        except Exception as e:
            logger.error(f"Insider trading analysis error: {e}")

        return pd.DataFrame(trades)

# ========================= Enhanced Stock Screener Integration =========================

class ComprehensiveMarketIntelligence:
    """Main intelligence system integrating all data sources"""

    def __init__(self, config: IntelligenceConfig):
        self.config = config

        # Initialize all modules
        self.sec_monitor = SECFilingMonitor(config)
        self.econ_tracker = EconomicCalendarTracker(config)
        self.congress_tracker = CongressionalTradeTracker(config)
        self.social_analyzer = SocialSentimentAnalyzer(config)
        self.institutional_tracker = InstitutionalTracker(config)
        self.insider_analyzer = InsiderTradingAnalyzer(config)

    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive market intelligence report"""
        logger.info("🚀 Starting comprehensive market intelligence scan...")

        # Gather all data sources concurrently
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(self.sec_monitor.get_recent_filings): 'sec_filings',
                executor.submit(self.econ_tracker.get_upcoming_events): 'economic_calendar',
                executor.submit(self.congress_tracker.get_recent_trades): 'congressional_trades',
                executor.submit(self.social_analyzer.get_trending_stocks): 'social_trends',
                executor.submit(self.institutional_tracker.get_hot_hands): 'institutional_activity',
                executor.submit(self.insider_analyzer.get_notable_insider_trades): 'insider_trades'
            }

            intelligence_data = {}
            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    result = future.result()
                    intelligence_data[data_type] = result
                    logger.success(f"✅ {data_type} data collected")
                except Exception as e:
                    logger.error(f"❌ {data_type} failed: {e}")
                    intelligence_data[data_type] = pd.DataFrame()

        # Get stock screening data
        stock_data = self._run_stock_screening()
        intelligence_data['stock_screening'] = stock_data

        # Generate recommendations
        recommendations = self._generate_recommendations(intelligence_data)

        # Create comprehensive report
        report = self._create_dashboard_report(intelligence_data, recommendations)

        # Save report
        self._save_report(report)

        return report

    def _run_stock_screening(self) -> pd.DataFrame:
        """Run basic stock screening (simplified version)"""
        logger.info("Running stock screening...")

        try:
            # Get S&P 500 tickers
            tickers = self._get_sp500_tickers()
            if self.config.tickers_limit:
                tickers = tickers[: self.config.tickers_limit]

            screening_data = []
            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    hist = stock.history(period="1y")

                    if len(hist) > 50:
                        current_price = hist['Close'].iloc[-1]

                        screening_data.append({
                            'ticker': ticker,
                            'company': info.get('shortName', ticker),
                            'sector': info.get('sector', 'Unknown'),
                            'price': current_price,
                            'market_cap': info.get('marketCap', 0),
                            'pe_ratio': info.get('trailingPE', 0),
                            'ret_1m': ((current_price / hist['Close'].iloc[-21]) - 1) * 100 if len(hist) > 21 else 0,
                            'ret_3m': ((current_price / hist['Close'].iloc[-63]) - 1) * 100 if len(hist) > 63 else 0,
                            'volume_avg': hist['Volume'].tail(20).mean(),
                            'rsi': self._calculate_rsi(hist['Close']).iloc[-1] if len(hist) > 14 else 50
                        })
                except Exception:
                    continue

            return pd.DataFrame(screening_data)

        except Exception as e:
            logger.error(f"Stock screening error: {e}")
            return pd.DataFrame()

    def _get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers with multiple robust fallbacks (like s&p500.ipynb)."""
        # 1) yfinance helper (newer versions)
        if hasattr(yf, "tickers_sp500"):
            try:
                syms = yf.tickers_sp500()
                if syms:
                    return [s.replace('.', '-').upper() for s in syms]
            except Exception:
                pass

        # 2) Wikipedia via pandas.read_html with alternate flavors
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        for flavor in (None, "bs4"):
            try:
                tbls = pd.read_html(wiki_url, match="Symbol", flavor=flavor)
                syms = tbls[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
                if len(syms) > 100:
                    return syms
            except Exception:
                pass

        # 3) CSV fallbacks
        for url in (
            "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
            "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        ):
            try:
                df = pd.read_csv(url)
                if "Symbol" in df.columns:
                    syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
                    if len(syms) > 100:
                        return syms
            except Exception:
                pass

        # Tiny fallback if everything fails
        logger.warning("Falling back to tiny S&P 500 subset universe.")
        return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "JPM", "XOM", "UNH", "AVGO", "LLY"]

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _generate_recommendations(self, intelligence_data: Dict) -> pd.DataFrame:
        """Generate stock recommendations based on all intelligence sources"""
        logger.info("Generating stock recommendations...")

        recommendations = []
        stock_data = intelligence_data.get('stock_screening', pd.DataFrame())

        if stock_data.empty:
            return pd.DataFrame()

        for _, stock in stock_data.iterrows():
            ticker = stock['ticker']

            # Calculate individual scores
            technical_score = self._calculate_technical_score(stock)
            fundamental_score = self._calculate_fundamental_score(stock)
            insider_score = self._calculate_insider_score(ticker, intelligence_data)
            institutional_score = self._calculate_institutional_score(ticker, intelligence_data)
            social_score = self._calculate_social_score(ticker, intelligence_data)
            macro_score = self._calculate_macro_score(stock, intelligence_data)
            congressional_score = self._calculate_congressional_score(ticker, intelligence_data)

            # Calculate composite score
            composite_score = (
                technical_score * self.config.technical_weight +
                fundamental_score * self.config.fundamental_weight +
                insider_score * self.config.insider_weight +
                institutional_score * self.config.institutional_weight +
                social_score * self.config.social_weight +
                macro_score * self.config.macro_weight +
                congressional_score * self.config.congressional_weight
            )

            # Generate recommendation
            if composite_score >= 80:
                recommendation = "STRONG BUY"
            elif composite_score >= 60:
                recommendation = "BUY"
            elif composite_score >= 40:
                recommendation = "HOLD"
            elif composite_score >= 20:
                recommendation = "SELL"
            else:
                recommendation = "STRONG SELL"

            recommendations.append({
                'ticker': ticker,
                'company': stock['company'],
                'sector': stock['sector'],
                'price': stock['price'],
                'recommendation': recommendation,
                'conviction_score': round(composite_score, 1),
                'technical_score': round(technical_score, 1),
                'fundamental_score': round(fundamental_score, 1),
                'insider_score': round(insider_score, 1),
                'institutional_score': round(institutional_score, 1),
                'social_score': round(social_score, 1),
                'macro_score': round(macro_score, 1),
                'congressional_score': round(congressional_score, 1),
                'key_catalysts': self._identify_catalysts(ticker, intelligence_data)
            })

        df = pd.DataFrame(recommendations)
        df = df[df['conviction_score'] >= self.config.min_conviction_score]
        return df.sort_values('conviction_score', ascending=False).head(self.config.max_recommendations)

    def _calculate_technical_score(self, stock: pd.Series) -> float:
        """Calculate technical analysis score"""
        score = 50  # Neutral base

        # RSI scoring
        rsi = stock.get('rsi', 50)
        if 40 <= rsi <= 60:
            score += 10  # Neutral zone
        elif 30 <= rsi < 40:
            score += 20  # Oversold
        elif rsi < 30:
            score += 30  # Very oversold
        elif 60 < rsi <= 70:
            score -= 10  # Overbought
        elif rsi > 70:
            score -= 20  # Very overbought

        # Momentum scoring
        ret_1m = stock.get('ret_1m', 0)
        ret_3m = stock.get('ret_3m', 0)

        if ret_1m > 5 and ret_3m > 10:
            score += 20  # Strong momentum
        elif ret_1m > 0 and ret_3m > 0:
            score += 10  # Positive momentum
        elif ret_1m < -10 or ret_3m < -20:
            score -= 20  # Weak momentum

        return max(0, min(100, score))

    def _calculate_fundamental_score(self, stock: pd.Series) -> float:
        """Calculate fundamental analysis score"""
        score = 50

        # P/E ratio scoring
        pe = stock.get('pe_ratio', 0)
        if 0 < pe <= 15:
            score += 20  # Cheap
        elif 15 < pe <= 25:
            score += 10  # Fair value
        elif pe > 40:
            score -= 20  # Expensive

        # Market cap (size factor)
        market_cap = stock.get('market_cap', 0)
        if market_cap > 100_000_000_000:  # Large cap
            score += 5  # Stability premium

        return max(0, min(100, score))

    def _calculate_insider_score(self, ticker: str, intelligence_data: Dict) -> float:
        """Calculate insider trading score"""
        insider_trades = intelligence_data.get('insider_trades', pd.DataFrame())

        if insider_trades.empty:
            return 50  # Neutral

        ticker_trades = insider_trades[insider_trades['ticker'] == ticker]
        if ticker_trades.empty:
            return 50

        # Simple scoring based on recent insider activity
        recent_trades = ticker_trades[ticker_trades['transaction'] == 'BUY']
        if len(recent_trades) > 0:
            return 80  # Bullish insider activity

        recent_sales = ticker_trades[ticker_trades['transaction'] == 'SELL']
        if len(recent_sales) > 2:  # Multiple recent sales
            return 30  # Bearish

        return 50

    def _calculate_institutional_score(self, ticker: str, intelligence_data: Dict) -> float:
        """Calculate institutional investor score"""
        institutional_data = intelligence_data.get('institutional_activity', pd.DataFrame())

        if institutional_data.empty:
            return 50

        ticker_activity = institutional_data[institutional_data['ticker'] == ticker]
        if ticker_activity.empty:
            return 50

        # Score based on smart money activity
        buy_activity = ticker_activity[ticker_activity['action'] == 'BUY']
        if len(buy_activity) > 0:
            return 75  # Smart money buying

        return 50

    def _calculate_social_score(self, ticker: str, intelligence_data: Dict) -> float:
        """Calculate social sentiment score"""
        social_data = intelligence_data.get('social_trends', pd.DataFrame())

        if social_data.empty:
            return 50

        ticker_trends = social_data[social_data['ticker'] == ticker]
        if ticker_trends.empty:
            return 50

        # Score based on social sentiment
        positive_trends = ticker_trends[ticker_trends['sentiment'] == 'POSITIVE']
        if len(positive_trends) > 0:
            avg_influence = positive_trends['influence_score'].mean()
            return min(90, 50 + avg_influence / 2)

        return 50

    def _calculate_macro_score(self, stock: pd.Series, intelligence_data: Dict) -> float:
        """Calculate macroeconomic environment score"""
        econ_events = intelligence_data.get('economic_calendar', pd.DataFrame())

        # Simple sector-based macro scoring
        sector = stock.get('sector', '')

        if 'Technology' in sector:
            return 60  # Generally favored in current environment
        elif 'Financial' in sector:
            return 65  # Benefits from higher rates
        elif 'Consumer' in sector:
            return 55  # Mixed signals

        return 50

    def _calculate_congressional_score(self, ticker: str, intelligence_data: Dict) -> float:
        """Calculate congressional trading score"""
        congress_data = intelligence_data.get('congressional_trades', pd.DataFrame())

        if congress_data.empty:
            return 50

        ticker_trades = congress_data[congress_data['ticker'] == ticker]
        if ticker_trades.empty:
            return 50

        # Score based on congressional activity
        buy_trades = ticker_trades[ticker_trades['transaction'] == 'BUY']
        if len(buy_trades) > 0:
            return 70  # Congressional buying

        return 50

    def _identify_catalysts(self, ticker: str, intelligence_data: Dict) -> str:
        """Identify key catalysts for the stock"""
        catalysts = []

        # Check insider trades
        insider_trades = intelligence_data.get('insider_trades', pd.DataFrame())
        ticker_insider = insider_trades[insider_trades['ticker'] == ticker]
        if not ticker_insider.empty:
            catalysts.append("Insider buying")

        # Check social trends
        social_trends = intelligence_data.get('social_trends', pd.DataFrame())
        ticker_social = social_trends[social_trends['ticker'] == ticker]
        if not ticker_social.empty:
            catalysts.append("Social media buzz")

        # Check institutional activity
        institutional = intelligence_data.get('institutional_activity', pd.DataFrame())
        ticker_inst = institutional[institutional['ticker'] == ticker]
        if not ticker_inst.empty:
            catalysts.append("Smart money activity")

        return "; ".join(catalysts) if catalysts else "Technical setup"

    def _create_dashboard_report(self, intelligence_data: Dict, recommendations: pd.DataFrame) -> Dict:
        """Create comprehensive dashboard report"""

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_stocks_analyzed': len(intelligence_data.get('stock_screening', [])),
                'recommendations_generated': len(recommendations),
                'strong_buys': len(recommendations[recommendations['recommendation'] == 'STRONG BUY']),
                'buys': len(recommendations[recommendations['recommendation'] == 'BUY']),
                'data_sources_active': len([k for k, v in intelligence_data.items() if not v.empty])
            },
            'market_intelligence': {
                'sec_filings': intelligence_data.get('sec_filings', pd.DataFrame()).to_dict('records'),
                'economic_calendar': intelligence_data.get('economic_calendar', pd.DataFrame()).to_dict('records'),
                'congressional_trades': intelligence_data.get('congressional_trades', pd.DataFrame()).to_dict('records'),
                'social_trends': intelligence_data.get('social_trends', pd.DataFrame()).to_dict('records'),
                'institutional_activity': intelligence_data.get('institutional_activity', pd.DataFrame()).to_dict('records'),
                'insider_trades': intelligence_data.get('insider_trades', pd.DataFrame()).to_dict('records')
            },
            'recommendations': recommendations.to_dict('records'),
            'top_picks': recommendations.head(5).to_dict('records') if not recommendations.empty else []
        }

        return report

    def _save_report(self, report: Dict):
        """Save report to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = self.config.output_dir / f"market_intelligence_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Save CSV recommendations
        csv_path = None
        if report['recommendations']:
            df = pd.DataFrame(report['recommendations'])
            csv_path = self.config.output_dir / f"recommendations_{timestamp}.csv"
            df.to_csv(csv_path, index=False)

        # Create HTML dashboard
        html_path = self.config.output_dir / f"dashboard_{timestamp}.html"
        self._create_html_dashboard(report, html_path)

        logger.success(f"📊 Reports saved:")
        logger.success(f"  • JSON: {json_path}")
        if csv_path is not None:
            logger.success(f"  • CSV: {csv_path}")
        logger.success(f"  • HTML: {html_path}")

    def _create_html_dashboard(self, report: Dict, html_path: Path):
        """Create HTML dashboard"""
        top_picks_html = ''.join([
            f"<p><strong>{pick['ticker']}</strong> - {pick['recommendation']} ({pick['conviction_score']}%)</p>"
            for pick in report.get('top_picks', [])[:3]
        ])

        recommendation_rows = ''.join([
            (
                '<tr class="' + rec['recommendation'].lower().replace(' ', '-') + '">\n'
                '    <td><strong>' + str(rec['ticker']) + '</strong></td>\n'
                '    <td>' + str(rec['company']) + '</td>\n'
                '    <td>' + str(rec['recommendation']) + '</td>\n'
                '    <td>' + str(rec['conviction_score']) + '%</td>\n'
                '    <td>$' + f"{float(rec['price']):.2f}" + '</td>\n'
                '    <td>' + str(rec['key_catalysts']) + '</td>\n'
                '</tr>'
            )
            for rec in report.get('recommendations', [])[:10]
        ])

        sec_list = ''.join([
            f"<li>{filing.get('company', 'Unknown')} - {filing.get('form_type', 'Unknown')}</li>"
            for filing in report.get('market_intelligence', {}).get('sec_filings', [])[:5]
        ])

        econ_list = ''.join([
            f"<li>{event.get('event', 'Unknown')} - {event.get('date', 'Unknown')}</li>"
            for event in report.get('market_intelligence', {}).get('economic_calendar', [])[:5]
        ])

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Market Intelligence Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                          gap: 15px; margin-bottom: 30px; }}
                .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .recommendation {{ border-left: 4px solid #4CAF50; margin-bottom: 15px; }}
                .strong-buy {{ border-left-color: #2196F3; }}
                .buy {{ border-left-color: #4CAF50; }}
                .hold {{ border-left-color: #FF9800; }}
                .sell {{ border-left-color: #f44336; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Market Intelligence Dashboard</h1>
                <p>Generated: {report['timestamp']}</p>
            </div>

            <div class="summary">
                <div class="card">
                    <h3>📊 Analysis Summary</h3>
                    <p><strong>Stocks Analyzed:</strong> {report['summary']['total_stocks_analyzed']}</p>
                    <p><strong>Recommendations:</strong> {report['summary']['recommendations_generated']}</p>
                    <p><strong>Strong Buys:</strong> {report['summary']['strong_buys']}</p>
                </div>
                <div class="card">
                    <h3>🎯 Top Picks</h3>
                    {top_picks_html}
                </div>
            </div>

            <div class="card">
                <h2>📈 Stock Recommendations</h2>
                <table>
                    <tr>
                        <th>Ticker</th>
                        <th>Company</th>
                        <th>Recommendation</th>
                        <th>Conviction</th>
                        <th>Price</th>
                        <th>Key Catalysts</th>
                    </tr>
                    {recommendation_rows}
                </table>
            </div>

            <div class="card">
                <h2>📋 Market Intelligence Sources</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h3>🏛️ SEC Filings</h3>
                        <ul>
                            {sec_list}
                        </ul>
                    </div>
                    <div>
                        <h3>📅 Economic Calendar</h3>
                        <ul>
                            {econ_list}
                        </ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        with open(html_path, 'w') as f:
            f.write(html_content)

# ========================= Main Execution =========================
def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive Market Intelligence System...")

    # Create configuration
    config = IntelligenceConfig(
        max_workers=8,
        min_conviction_score=50.0,
        max_recommendations=15
    )

    # Initialize system
    intelligence = ComprehensiveMarketIntelligence(config)

    try:
        # Generate comprehensive report
        report = intelligence.generate_comprehensive_report()

        print("\n" + "="*80)
        print("📊 MARKET INTELLIGENCE DASHBOARD SUMMARY")
        print("="*80)

        print(f"📈 Total Stocks Analyzed: {report['summary']['total_stocks_analyzed']}")
        print(f"🎯 Recommendations Generated: {report['summary']['recommendations_generated']}")
        print(f"⭐ Strong Buys: {report['summary']['strong_buys']}")
        print(f"📊 Data Sources Active: {report['summary']['data_sources_active']}")

        if report['top_picks']:
            print("\n🏆 TOP PICKS:")
            for i, pick in enumerate(report['top_picks'][:5], 1):
                print(f"  {i}. {pick['ticker']} - {pick['recommendation']} "
                      f"(Conviction: {pick['conviction_score']}%)")
                print(f"     {pick['company']} | Catalysts: {pick['key_catalysts']}")

        print(f"\n📁 Full dashboard saved to: {config.output_dir}")
        print("✅ Market Intelligence System Complete!")

    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
