#!/usr/bin/env python3
"""
Plot a line chart of 50m wind speed from data/single.csv.

CSV schema:
  YEAR, MO, DY, WS50M

Usage:
  python plot_single_windspeed.py [--input data/single.csv] [--output figures/single_windspeed.png] [--show]

By default, saves a PNG to figures/ and does not display a window. Use --show to display.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 50m wind speed time series from CSV")
    parser.add_argument(
        "--input",
        default="data/single.csv",
        help="Path to input CSV with columns YEAR,MO,DY,WS50M (default: data/single.csv)",
    )
    parser.add_argument(
        "--output",
        default="figures/single_windspeed.png",
        help="Path to save the output PNG (default: figures/single_windspeed.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window (in addition to saving)",
    )
    return parser.parse_args()


def read_data(csv_path: Path) -> pd.DataFrame:
    # Combine YEAR, MO, DY into a single datetime column via parse_dates
    df = pd.read_csv(
        csv_path,
        parse_dates={"date": ["YEAR", "MO", "DY"]},
        keep_date_col=False,
    )
    # Ensure expected column exists
    if "WS50M" not in df.columns:
        raise ValueError("Input CSV must contain a 'WS50M' column")
    # Sort by date just in case and set index for nicer plotting
    df = df.sort_values("date").set_index("date", drop=True)
    return df


def plot(df: pd.DataFrame, out_path: Path, show: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(df.index, df["WS50M"], color="#1f77b4", linewidth=0.8)

    # Axis labels and title (ASCII to avoid font issues)
    ax.set_title("50m Wind Speed Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("WS50M")

    # Use concise date formatter for long ranges
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    csv_path = Path(args.input)
    out_path = Path(args.output)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = read_data(csv_path)
    plot(df, out_path, show=args.show)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
