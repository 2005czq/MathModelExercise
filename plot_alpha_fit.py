#!/usr/bin/env python3
"""
Visualize the wind shear fit using v_H = v_h * (H/h)^alpha with H=50m, h=10m.

Reads data/point.csv (NASA POWER format with header block), computes the global
alpha = mean( ln(WS50M/WS10M) ) / ln(5), and plots:

- Scatter: WS10M (x) vs WS50M (y) for a subsample of points
- Fitted line: y = x * 5^alpha across the x-range

Outputs: figures/alpha_fit.png
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


DEFAULT_INPUT = Path(__file__).parent / "data" / "point.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "figures" / "alpha_fit.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot WS10M vs WS50M with fitted alpha line")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to data/point.csv")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to save the PNG")
    p.add_argument(
        "--sample",
        type=int,
        default=20000,
        help="Max number of points to scatter plot (subsample if larger)",
    )
    p.add_argument("--show", action="store_true", help="Show the figure window")
    return p.parse_args()


def read_valid_speeds(path: Path) -> Tuple[List[float], List[float]]:
    """Return lists of (ws10, ws50) filtering header, missing (-999), and non-positive."""
    ws10: List[float] = []
    ws50: List[float] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            if not in_data:
                if row[0].strip().upper() == "YEAR":
                    in_data = True
                continue
            if len(row) < 6:
                continue
            try:
                v50 = float(row[4])
                v10 = float(row[5])
            except ValueError:
                continue
            if v50 == -999 or v10 == -999:
                continue
            if v50 <= 0.0 or v10 <= 0.0:
                continue
            ws10.append(v10)
            ws50.append(v50)
    return ws10, ws50


def fit_alpha(ws10: List[float], ws50: List[float]) -> float:
    """Compute alpha = mean(ln(ws50/ws10))/ln(5)."""
    ln5 = math.log(5.0)
    ratios = [math.log(v50 / v10) for v10, v50 in zip(ws10, ws50) if v50 > 0 and v10 > 0]
    if not ratios:
        raise ValueError("No valid ratios to fit alpha")
    return sum(ratios) / len(ratios) / ln5


def subsample(xs: List[float], ys: List[float], k: int) -> Tuple[List[float], List[float]]:
    n = len(xs)
    if n <= k:
        return xs, ys
    step = max(1, n // k)
    xs_s = xs[::step][:k]
    ys_s = ys[::step][:k]
    return xs_s, ys_s


def plot_fit(ws10: List[float], ws50: List[float], alpha: float, out_path: Path, show: bool, max_points: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs, ys = subsample(ws10, ws50, max_points)

    # Range for the fitted line
    xmin = min(xs)
    xmax = max(xs)
    x_line = [xmin, xmax]
    scale = 5.0 ** alpha
    y_line = [x * scale for x in x_line]

    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    ax.scatter(xs, ys, s=2, alpha=0.3, color="#1f77b4", edgecolors="none", label="Observed")
    ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label=f"Fit: WS50 = WS10 * 5^alpha\nalpha = {alpha:.4f}")

    ax.set_xlabel("WS10M (m/s)")
    ax.set_ylabel("WS50M (m/s)")
    ax.set_title("Wind Shear Fit: WS50M vs WS10M")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    ws10, ws50 = read_valid_speeds(in_path)
    alpha = fit_alpha(ws10, ws50)
    print(f"Fitted alpha: {alpha:.6f} (based on {len(ws10)} samples)")
    plot_fit(ws10, ws50, alpha, out_path, args.show, args.sample)
    print(f"Saved fit plot to: {out_path}")


if __name__ == "__main__":
    main()
