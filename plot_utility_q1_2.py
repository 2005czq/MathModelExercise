#!/usr/bin/env python3
"""
Compute utility = average power / cost for turbine configurations (D,H)
using hourly WS50M data, extrapolating to hub height H, and plot a grouped
bar chart by D with three bars per group for H.

Data source: data/point.csv (NASA POWER format with header block).
Output figure: figures/utility_q1_2_grouped.png
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "point.csv")
FIG_PATH = os.path.join(os.path.dirname(__file__), "figures", "utility_q1_2_grouped.png")


def read_ws50_series(csv_path: str) -> List[float]:
    """Read WS50M hourly series from NASA POWER CSV with leading header.

    Returns a list of floats (m/s), skipping missing (-999) values.
    """
    ws50: List[float] = []
    header_found = False
    with open(csv_path, "r", encoding="utf-8") as f:
      for line in f:
        s = line.strip()
        if not header_found:
            if s.startswith("YEAR,MO,DY,HR,WS50M,WS10M"):
                header_found = True
            # continue to next line until header
            continue
        if not s or s.startswith("#"):
            continue
        parts = s.split(",")
        if len(parts) < 6:
            continue
        try:
            v = float(parts[4])
        except ValueError:
            continue
        if v == -999:
            continue
        ws50.append(v)
    if not ws50:
        raise RuntimeError("No WS50M data parsed from CSV. Check file format.")
    return ws50


def wind_at_height(v50: float, H: float) -> float:
    """Extrapolate wind speed at hub height H from 50m using given power law."""
    return v50 * (H / 50.0) ** 0.073096


def power_mw(v: float, D: float) -> float:
    """Piecewise power curve P(v) in MW.

    P(v)=
    0,  v<2.5
    0.191619426414177*D^2*v^3,  2.5<=v<10
    6,  10<=v<=24
    0,  v>24
    """
    if v < 2.5:
        return 0.0
    if v < 10.0:
        return 0.191619426414177 * (D ** 2) * (v ** 3)
    if v <= 24.0:
        return 6.0
    return 0.0


def cost(D: float, H: float) -> float:
    """Cost approximation using provided weights: C = w1*D^2 + w2*H + w3."""
    w1 = 0.1557211415
    w2 = 15.6249220271
    w3 = -2065.5926440647
    return w1 * (D ** 2) + w2 * H + w3


def compute_mean_power(ws50_series: List[float], D: float, H: float) -> float:
    total = 0.0
    for v50 in ws50_series:
        vH = wind_at_height(v50, H)
        total += power_mw(vH, D)
    return total / len(ws50_series)


def main() -> None:
    # Define configurations
    Ds = [80.0, 100.0, 120.0]
    Hs = [90.0, 120.0, 150.0]

    # Load data
    ws50_series = read_ws50_series(DATA_PATH)

    # Compute utility for each (D,H)
    mean_power: Dict[Tuple[float, float], float] = {}
    util: Dict[Tuple[float, float], float] = {}
    for D in Ds:
        for H in Hs:
            mp = compute_mean_power(ws50_series, D, H)
            c = cost(D, H)
            mean_power[(D, H)] = mp
            # Handle zero cost to avoid division by zero
            util[(D, H)] = mp / c if c != 0 else float("inf")

    # Print a compact table
    print("D,H,mean_power_MW,cost,utility(power/cost)")
    for D in Ds:
        for H in Hs:
            mp = mean_power[(D, H)]
            c = cost(D, H)
            u = util[(D, H)]
            print(f"{int(D)},{int(H)},{mp:.6f},{c:.6f},{u:.8f}")

    # Plot grouped bar chart: groups by H, bars for D
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x_positions = list(range(len(Hs)))
    width = 0.22
    offsets = [-width, 0.0, width]
    colors = ["#4e79a7", "#59a14f", "#e15759"]

    for idx_d, (D, offset, color) in enumerate(zip(Ds, offsets, colors)):
        heights = [util[(D, H)] for H in Hs]
        bars = ax.bar([x + offset for x in x_positions], heights, width=width, color=color, label=f"D={int(D)}m")
        # Annotate values on each bar
        for b in bars:
            h = b.get_height()
            va = "bottom" if h >= 0 else "top"
            y = h + (0.02 * abs(h) + 1e-6) if h >= 0 else h - (0.02 * abs(h) + 1e-6)
            ax.text(b.get_x() + b.get_width()/2, y, f"{h:.3f}", ha="center", va=va, fontsize=9, rotation=0)

    ax.set_xticks(x_positions, [f"H={int(H)}m" for H in Hs])
    ax.set_ylabel("Utility = Mean Power (MW) / Cost C(D,H)")
    ax.set_title("Utility by Hub Height (groups) and Diameter (bars)")
    ax.legend(title="Diameter")
    ax.axhline(0, color="#333", linewidth=0.8)
    fig.tight_layout()

    # Ensure figures directory exists
    os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Saved grouped bar chart to: {FIG_PATH}")


if __name__ == "__main__":
    main()
