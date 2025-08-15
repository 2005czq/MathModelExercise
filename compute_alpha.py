#!/usr/bin/env python3
"""
Compute wind shear exponent alpha using v_H = v_h * (H/h)^alpha
from hourly NASA POWER MERRA-2 wind speeds at 50m (WS50M) and 10m (WS10M).

Input CSV: data/point.csv
- Contains a header block, then a line: YEAR,MO,DY,HR,WS50M,WS10M
- Missing values are -999 per header notes.

Output:
- Prints overall fitted alpha (mean of log-ratios) and basic stats.
- Writes per-hour alpha to data/alpha_estimates.csv with the timestamp columns.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, median, pstdev


DATA_PATH = Path(__file__).parent / "data" / "point.csv"
OUT_PATH = Path(__file__).parent / "data" / "alpha_estimates.csv"


def parse_rows(csv_path: Path):
    """Yield rows of (year, mo, dy, hr, ws50, ws10) from the CSV, skipping header block."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            # Detect the data header line
            if not in_data:
                if row[0].strip().upper() == "YEAR":
                    in_data = True
                continue
            # Expect 6 columns thereafter
            if len(row) < 6:
                continue
            try:
                y = int(row[0])
                mo = int(row[1])
                dy = int(row[2])
                hr = int(row[3])
                ws50 = float(row[4])
                ws10 = float(row[5])
            except ValueError:
                continue
            yield y, mo, dy, hr, ws50, ws10


def compute_alpha_values(rows, min_ws: float = 1e-6):
    """
    Compute per-sample alpha_i = ln(WS50M/WS10M) / ln(50/10) for valid rows.

    Filters:
    - Exclude missing values (-999) and non-positive speeds.
    - Enforce a small minimum wind speed to avoid extreme ratios.
    """
    ln_ratio_denom = math.log(50.0 / 10.0)  # ln(5)
    for y, mo, dy, hr, ws50, ws10 in rows:
        if ws50 <= 0 or ws10 <= 0:
            continue
        if ws50 == -999 or ws10 == -999:
            continue
        if ws50 < min_ws or ws10 < min_ws:
            continue
        alpha_i = math.log(ws50 / ws10) / ln_ratio_denom
        yield (y, mo, dy, hr, alpha_i)


def summarize(alphas):
    """Return basic stats for a sequence of alpha values."""
    arr = list(alphas)
    if not arr:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    def percentile(p: float) -> float:
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return arr_sorted[lo]
        w = idx - lo
        return arr_sorted[lo] * (1 - w) + arr_sorted[hi] * w

    return {
        "count": n,
        "mean": mean(arr_sorted),
        "median": median(arr_sorted),
        "std": pstdev(arr_sorted) if n > 1 else 0.0,
        "p05": percentile(0.05),
        "p95": percentile(0.95),
    }


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Input not found: {DATA_PATH}")

    rows = list(parse_rows(DATA_PATH))
    alpha_rows = list(compute_alpha_values(rows))

    # Write per-hour alpha file
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["YEAR", "MO", "DY", "HR", "ALPHA"])
        for y, mo, dy, hr, a in alpha_rows:
            w.writerow([y, mo, dy, hr, f"{a:.6f}"])

    # Summarize
    alpha_values = [a for *_t, a in alpha_rows]
    stats = summarize(alpha_values)

    # Print summary
    print("Fitted wind shear exponent alpha from WS50M and WS10M")
    print(f"Samples used: {stats['count']}")
    print(f"alpha (mean of ln ratios): {stats['mean']:.6f}")
    print(f"median: {stats['median']:.6f}")
    print(f"std (population): {stats['std']:.6f}")
    print(f"p05..p95: {stats['p05']:.6f} .. {stats['p95']:.6f}")
    print(f"Per-hour estimates written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
