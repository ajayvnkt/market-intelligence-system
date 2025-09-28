#!/usr/bin/env python3
"""
Quick Start Example for Market Intelligence System

This script demonstrates the basic usage of the Market Intelligence System
with different investment strategy configurations.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from comprehensive_market_intelligence import ComprehensiveMarketIntelligence, IntelligenceConfig


def conservative_strategy():
    """Example: Conservative investment strategy"""
    print("🔧 Running Conservative Investment Strategy...")
    
    config = IntelligenceConfig(
        max_workers=4,
        min_conviction_score=70.0,  # Higher threshold for conservative approach
        max_recommendations=8,      # Fewer, higher-quality picks
        technical_weight=0.20,      # Lower technical weight
        fundamental_weight=0.30,    # Higher fundamental weight
        insider_weight=0.20,        # Higher insider weight
        institutional_weight=0.20,  # Higher institutional weight
        social_weight=0.05,         # Lower social weight
        macro_weight=0.05
    )
    
    engine = ComprehensiveMarketIntelligence(config)
    report = engine.generate_comprehensive_report()
    
    print(f"✅ Conservative Analysis Complete:")
    print(f"   📊 Stocks Analyzed: {report['summary']['total_stocks_analyzed']}")
    print(f"   🎯 Recommendations: {report['summary']['recommendations_generated']}")
    print(f"   ⭐ Strong Buys: {report['summary']['strong_buys']}")
    
    return report


def growth_strategy():
    """Example: Growth investment strategy"""
    print("🔧 Running Growth Investment Strategy...")
    
    config = IntelligenceConfig(
        max_workers=8,
        min_conviction_score=55.0,  # Lower threshold for more opportunities
        max_recommendations=15,     # More picks for diversification
        technical_weight=0.30,      # Higher technical weight
        fundamental_weight=0.15,    # Lower fundamental weight
        insider_weight=0.10,        # Lower insider weight
        institutional_weight=0.25,  # Higher institutional weight
        social_weight=0.15,         # Higher social weight
        macro_weight=0.05
    )
    
    engine = ComprehensiveMarketIntelligence(config)
    report = engine.generate_comprehensive_report()
    
    print(f"✅ Growth Analysis Complete:")
    print(f"   📊 Stocks Analyzed: {report['summary']['total_stocks_analyzed']}")
    print(f"   🎯 Recommendations: {report['summary']['recommendations_generated']}")
    print(f"   ⭐ Strong Buys: {report['summary']['strong_buys']}")
    
    return report


def social_focused_strategy():
    """Example: Social media focused strategy"""
    print("🔧 Running Social Media Focused Strategy...")
    
    config = IntelligenceConfig(
        max_workers=6,
        min_conviction_score=60.0,
        max_recommendations=12,
        technical_weight=0.25,      # Technical confirmation
        fundamental_weight=0.15,    # Lower fundamental weight
        insider_weight=0.05,        # Lower insider weight
        institutional_weight=0.20,  # Smart money validation
        social_weight=0.30,         # High social weight
        macro_weight=0.05
    )
    
    engine = ComprehensiveMarketIntelligence(config)
    report = engine.generate_comprehensive_report()
    
    print(f"✅ Social Media Analysis Complete:")
    print(f"   📊 Stocks Analyzed: {report['summary']['total_stocks_analyzed']}")
    print(f"   🎯 Recommendations: {report['summary']['recommendations_generated']}")
    print(f"   ⭐ Strong Buys: {report['summary']['strong_buys']}")
    
    return report


def main():
    """Main function to run examples"""
    print("🚀 Market Intelligence System - Quick Start Examples")
    print("=" * 60)
    
    try:
        # Run different strategies
        print("\n1. Conservative Strategy")
        conservative_report = conservative_strategy()
        
        print("\n2. Growth Strategy")
        growth_report = growth_strategy()
        
        print("\n3. Social Media Focused Strategy")
        social_report = social_focused_strategy()
        
        # Display top picks from each strategy
        print("\n" + "=" * 60)
        print("🏆 TOP PICKS COMPARISON")
        print("=" * 60)
        
        strategies = [
            ("Conservative", conservative_report),
            ("Growth", growth_report),
            ("Social Media", social_report)
        ]
        
        for strategy_name, report in strategies:
            if report.get('top_picks'):
                print(f"\n{strategy_name} Strategy Top Picks:")
                for i, pick in enumerate(report['top_picks'][:3], 1):
                    print(f"  {i}. {pick['ticker']} - {pick['recommendation']} "
                          f"(Conviction: {pick['conviction_score']}%)")
        
        print(f"\n📁 All reports saved to: market_intelligence/")
        print("✅ Quick Start Examples Complete!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
