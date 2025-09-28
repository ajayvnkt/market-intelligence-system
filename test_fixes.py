#!/usr/bin/env python3
"""
Test script to verify the fixes work
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from comprehensive_market_intelligence import ComprehensiveMarketIntelligence, IntelligenceConfig

def test_system():
    """Test the market intelligence system with minimal configuration"""
    print("🧪 Testing Market Intelligence System...")
    
    # Configure with minimal settings for testing
    config = IntelligenceConfig(
        max_workers=2,                    # Reduced for testing
        min_conviction_score=30.0,        # Lower threshold for testing
        max_recommendations=3,            # Minimal recommendations
        tickers_limit=5,                  # Test with only 5 tickers
        output_dir=Path("test_output")
    )
    
    print("🚀 Initializing Market Intelligence Engine...")
    engine = ComprehensiveMarketIntelligence(config)
    
    print("📊 Starting test analysis...")
    try:
        report = engine.generate_comprehensive_report()
        
        print("\n" + "="*50)
        print("✅ TEST RESULTS")
        print("="*50)
        print(f"📊 Stocks Analyzed: {report['summary']['total_stocks_analyzed']}")
        print(f"🎯 Recommendations Generated: {report['summary']['recommendations_generated']}")
        print(f"⭐ Strong Buys: {report['summary']['strong_buys']}")
        print(f"📈 Buy Recommendations: {report['summary']['buys']}")
        print(f"🔍 Data Sources Active: {report['summary']['data_sources_active']}")
        
        if report.get('top_picks'):
            print("\n🏆 TOP PICKS:")
            for i, pick in enumerate(report['top_picks'][:3], 1):
                print(f"  {i}. {pick['ticker']} - {pick['recommendation']} "
                      f"(Conviction: {pick['conviction_score']}%)")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
