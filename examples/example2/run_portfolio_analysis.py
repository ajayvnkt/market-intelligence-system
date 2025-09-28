"""Run a focused market intelligence analysis for a custom portfolio watchlist."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.comprehensive_market_intelligence import (
    ComprehensiveMarketIntelligence,
    IntelligenceConfig,
)


def _load_watchlist(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Watchlist file {csv_path} has no rows to analyse.")

    candidates = [
        col
        for col in df.columns
        if str(col).strip().lower() in {"ticker", "symbol", "tickers", "symbols"}
    ]

    if candidates:
        series = df[candidates[0]]
    else:
        series = df.iloc[:, 0]

    tickers = (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)
    )

    tickers = [ticker for ticker in tickers.tolist() if ticker]
    if not tickers:
        raise ValueError(
            "No valid ticker symbols were found in the watchlist."
        )
    return tickers


def _format_price(value: float | int | None) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    if pd.isna(numeric):
        return "—"
    return f"${numeric:,.2f}"


def _render_summary(rows: Iterable[dict]) -> str:
    if not rows:
        return "No actionable insights were generated."

    header = [
        "Ticker",
        "Company",
        "Signal",
        "Conviction",
        "Entry",
        "Stop",
        "Target",
        "Exit Plan",
    ]
    divider = "-" * 120
    lines = [divider, " | ".join(header), divider]

    for row in rows:
        lines.append(
            " | ".join(
                [
                    row["ticker"],
                    row.get("company", "—"),
                    row.get("recommendation", "—"),
                    f"{row.get('conviction_score', 0):.1f}%"
                    if row.get("conviction_score") is not None
                    else "—",
                    _format_price(row.get("entry_price")),
                    _format_price(row.get("stop_loss_price")),
                    _format_price(row.get("target_price")),
                    row.get("exit_plan", "—"),
                ]
            )
        )
    lines.append(divider)
    return "\n".join(lines)


def run_portfolio_analysis(csv_path: Path, output_dir: Path) -> dict:
    tickers = _load_watchlist(csv_path)

    config = IntelligenceConfig(
        max_workers=min(4, max(1, len(tickers))),
        min_conviction_score=0.0,
        max_recommendations=len(tickers),
        min_price=0,
        min_volume=0,
        tickers_limit=len(tickers),
        custom_tickers=tickers,
        output_dir=output_dir,
    )

    engine = ComprehensiveMarketIntelligence(config)
    report = engine.generate_comprehensive_report()

    screening_records = report.get("market_intelligence", {}).get("stock_screening", [])
    screening_df = pd.DataFrame(screening_records)
    recommendation_df = pd.DataFrame(report.get("recommendations", []))
    recommendation_map = {
        row["ticker"]: row for row in recommendation_df.to_dict("records")
    }

    summary_rows: List[dict] = []
    for ticker in tickers:
        summary = {"ticker": ticker}
        if not screening_df.empty:
            match = screening_df[screening_df["ticker"] == ticker]
            if not match.empty:
                summary.update(match.iloc[0].to_dict())
        if ticker in recommendation_map:
            summary.update(recommendation_map[ticker])
        summary_rows.append(summary)

    summary_text = _render_summary(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "portfolio_summary.txt"
    output_file.write_text(summary_text)

    print("\n" + summary_text + "\n")
    print(f"Detailed artifacts saved to: {output_dir.resolve()}")

    return {
        "tickers": tickers,
        "summary_rows": summary_rows,
        "summary_path": output_file,
        "report": report,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the market intelligence engine for a specific set of tickers "
            "defined in a CSV watchlist."
        )
    )
    default_csv = Path(__file__).resolve().parent / "input_portfolio.csv"
    default_output = Path.cwd() / "outputs" / "example2"
    parser.add_argument(
        "--input",
        type=Path,
        default=default_csv,
        help="Path to the CSV file that contains the ticker symbols to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory where the generated summary and artifacts will be stored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_portfolio_analysis(args.input, args.output_dir)
