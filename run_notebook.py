#!/usr/bin/env python3
"""
Script to run the comprehensive market intelligence notebook with error handling
"""

import math
import sys
from pathlib import Path
import traceback

# Add src to path
current_dir = Path.cwd()
src_dir = current_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

def run_notebook_cells():
    """Run the notebook cells with error handling"""
    print("🚀 Running Market Intelligence Notebook...")
    
    try:
        # Cell 1: Environment Setup
        print("\n📋 Cell 1: Environment Setup")
        import os
        import sys
        from pathlib import Path

        # Load environment variables from .env file if it exists
        env_path = Path(".env")
        if env_path.exists():
            print("Loading environment variables from .env file...")
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
            print("✅ Environment variables loaded")
        else:
            print("ℹ️ No .env file found - using default configuration")

        # Configure S&P 500 analysis scope
        os.environ.pop("CMI_TICKERS_LIMIT", None)
        print("📊 Configured for full S&P 500 analysis")

        # Add current directory to Python path for imports
        current_dir = Path.cwd()
        if str(current_dir) not in sys.path:
            sys.path.append(str(current_dir))

        # Add src directory to path
        src_dir = current_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.append(str(src_dir))

        # Import the market intelligence module
        try:
            import comprehensive_market_intelligence as cmi
            print("✅ Market Intelligence module imported successfully")
        except ImportError as e:
            print(f"❌ Error importing module: {e}")
            return False

        # Cell 2: Market Intelligence Analysis Pipeline
        print("\n📊 Cell 2: Market Intelligence Analysis Pipeline")
        from importlib import reload
        from pathlib import Path
        import time

        # Reload module to ensure latest changes
        reload(cmi)

        # Configure the intelligence system with conservative settings
        config = cmi.IntelligenceConfig(
            max_workers=4,                    # Reduced for stability
            min_conviction_score=40.0,        # Lower threshold for more results
            max_recommendations=8,            # Reasonable number of recommendations
            tickers_limit=20,                 # Limit for faster execution
            output_dir=Path("market_intelligence")  # Output directory
        )

        print("🚀 Initializing Market Intelligence Engine...")
        engine = cmi.ComprehensiveMarketIntelligence(config)

        print("📊 Starting comprehensive market analysis...")
        start_time = time.time()

        # Generate the comprehensive report
        report = engine.generate_comprehensive_report()

        end_time = time.time()
        analysis_time = end_time - start_time

        # Display summary results
        print("\n" + "="*60)
        print("📈 MARKET INTELLIGENCE ANALYSIS COMPLETE")
        print("="*60)
        print(f"⏱️  Analysis Time: {analysis_time:.1f} seconds")
        print(f"📊 Stocks Analyzed: {report['summary']['total_stocks_analyzed']}")
        print(f"🎯 Recommendations Generated: {report['summary']['recommendations_generated']}")
        print(f"⭐ Strong Buys: {report['summary']['strong_buys']}")
        print(f"📈 Buy Recommendations: {report['summary']['buys']}")
        print(f"🔍 Data Sources Active: {report['summary']['data_sources_active']}")

        # Display top picks
        top_picks = report.get('top_picks') or []
        if top_picks:
            print("\n🏆 TOP RECOMMENDATIONS:")
            for i, pick in enumerate(top_picks[:5], 1):
                print(
                    f"  {i}. {pick['ticker']} - {pick['recommendation']} "
                    f"(Conviction: {pick['conviction_score']}%)"
                )
                print(f"     {pick['company']} | Catalysts: {pick['key_catalysts']}")

                entry = pick.get('entry_price') or pick.get('price')
                stop = pick.get('stop_loss_price')
                target = pick.get('target_price')
                drivers = pick.get('technical_drivers')
                plan = pick.get('exit_plan') or pick.get('exit_strategy')

                def _fmt(value: object) -> str:
                    if value is None:
                        return "—"
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        return "—"
                    if math.isnan(numeric):
                        return "—"
                    return f"${numeric:.2f}"

                print(
                    f"     Entry { _fmt(entry) } | Stop { _fmt(stop) } | Target { _fmt(target) } "
                    f"| Hold {pick.get('exit_review_days') or pick.get('holding_period_days', '—')} days"
                )
                if drivers:
                    print(f"     Technical: {drivers}")
                if plan:
                    print(f"     Exit plan: {plan}")

        # Display generated artifacts (JSON/CSV/HTML)
        artifacts = {
            name: Path(path) for name, path in (report.get('artifacts') or {}).items()
        }
        if artifacts:
            print("\n📦 Generated artifacts:")
            for label, path in artifacts.items():
                pretty = label.upper()
                print(f"  • {pretty}: {path.resolve() if path else path}")

            html_path = artifacts.get('html')
            if html_path and html_path.exists():
                print(f"\n📊 Dashboard generated: {html_path.name}")
                print(f"📁 Open this file in your browser: {html_path.resolve()}")
            else:
                print("\n⚠️ Dashboard artifact missing — check logs for details.")
        else:
            out_dir = config.output_dir
            print("\n⚠️ No artifacts were reported by the engine.")
            print(f"   Expected outputs to appear under: {Path(out_dir).resolve()}")

        print(f"\n📁 All reports saved to: {Path(config.output_dir).resolve()}")
        print("✅ Market Intelligence Analysis Complete!")
        
        return True

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("\n🔍 Full error traceback:")
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("📊 Market Intelligence System - Notebook Runner")
    print("=" * 60)
    
    success = run_notebook_cells()
    
    if success:
        print("\n🎉 Notebook execution completed successfully!")
        print("📁 Check the 'market_intelligence' folder for generated reports")
    else:
        print("\n💥 Notebook execution failed")
        print("🔧 Check the error messages above for troubleshooting")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
