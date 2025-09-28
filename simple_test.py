#!/usr/bin/env python3
"""
Simple test to verify basic functionality
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path.cwd()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

def test_imports():
    """Test that all imports work"""
    try:
        from comprehensive_market_intelligence import ComprehensiveMarketIntelligence, IntelligenceConfig
        print("✅ Imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration creation"""
    try:
        from comprehensive_market_intelligence import IntelligenceConfig
        config = IntelligenceConfig(
            max_workers=1,
            min_conviction_score=50.0,
            max_recommendations=1,
            tickers_limit=1
        )
        print("✅ Configuration created successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_engine_creation():
    """Test engine creation"""
    try:
        from comprehensive_market_intelligence import ComprehensiveMarketIntelligence, IntelligenceConfig
        config = IntelligenceConfig(
            max_workers=1,
            min_conviction_score=50.0,
            max_recommendations=1,
            tickers_limit=1
        )
        engine = ComprehensiveMarketIntelligence(config)
        print("✅ Engine created successfully")
        return True
    except Exception as e:
        print(f"❌ Engine creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Simple Tests...")
    
    tests = [
        test_imports,
        test_config,
        test_engine_creation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✅ All basic tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
