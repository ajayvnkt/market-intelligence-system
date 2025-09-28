"""
Market Intelligence System

A comprehensive market intelligence platform that integrates multiple data sources
to generate actionable stock recommendations using advanced analytics and real-time market data.
"""

__version__ = "1.0.0"
__author__ = "Ajay VNKT"
__email__ = "ajayvnkt@example.com"

from .comprehensive_market_intelligence import (
    ComprehensiveMarketIntelligence,
    IntelligenceConfig,
    SECFilingMonitor,
    EconomicCalendarTracker,
    CongressionalTradeTracker,
    SocialSentimentAnalyzer,
    InstitutionalTracker,
    InsiderTradingAnalyzer,
)

__all__ = [
    "ComprehensiveMarketIntelligence",
    "IntelligenceConfig", 
    "SECFilingMonitor",
    "EconomicCalendarTracker",
    "CongressionalTradeTracker",
    "SocialSentimentAnalyzer",
    "InstitutionalTracker",
    "InsiderTradingAnalyzer",
]
