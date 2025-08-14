#!/usr/bin/env python3
"""
Extract the last decade's windspeed time series at a specific observation point
from data/windspeed.csv into a separate CSV.

Input CSV format (header present):
LAT,LON,YEAR,MO,DY,WS50M

Target point: (25.00000 N, 120.00000 E)
Date range: 2015-08-01 through 2025-08-01 inclusive

Output written to: data/windspeed_25.0N_120.0E_2015-08-01_2025-08-01.csv
"""

from __future__ import annotations

import csv
import os
from typing import Tuple


IN_PATH = os.path.join("data", "windspeed.csv")
OUT_PATH = os.path.join(
    "data", "windspeed_25.0N_120.0E_2015-08-01_2025-08-01.csv"
)

# Target lat/lon and comparison tolerance to avoid float text quirks
TARGET_LAT = 25.0
TARGET_LON = 120.0
TOL = 1e-6

# Inclusive date range as (Y, M, D) tuples
START = (2015, 8, 1)
END = (2025, 8, 1)


def within_tolerance(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def within_range(ymd: Tuple[int, int, int], start: Tuple[int, int, int], end: Tuple[int, int, int]) -> bool:
    return start <= ymd <= end


def extract_point_timeseries(in_path: str = IN_PATH, out_path: str = OUT_PATH) -> int:
    """Extract rows for the target point and date range. Returns number of rows written."""
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    written = 0
    with open(in_path, newline="", encoding="utf-8") as f_in, open(
        out_path, "w", newline="", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames or ["LAT", "LON", "YEAR", "MO", "DY", "WS50M"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            try:
                lat = float(row["LAT"])  # type: ignore[index]
                lon = float(row["LON"])  # type: ignore[index]
                year = int(row["YEAR"])  # type: ignore[index]
                mo = int(row["MO"])  # type: ignore[index]
                dy = int(row["DY"])  # type: ignore[index]
            except (KeyError, ValueError):
                # Skip malformed rows
                continue

            if not (within_tolerance(lat, TARGET_LAT) and within_tolerance(lon, TARGET_LON)):
                continue

            if not within_range((year, mo, dy), START, END):
                continue

            writer.writerow(row)
            written += 1

    return written


if __name__ == "__main__":
    n = extract_point_timeseries()
    print(f"Wrote {n} rows to {OUT_PATH}")
