import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse geometry and IO from Q2.1
from optimize_layout_q2_1 import (
    L,
    ForbiddenLines,
    load_hourly_ws50,
    best_hex_grid_in_allowed,
    visualize_layout,
)


def compute_power_series(ws50: pd.DataFrame, H: float, D: float,
                         start: str, end: str) -> pd.Series:
    """Per-turbine power time series (MW) for the given window.

    Uses piecewise power curve with v(H) from WS50M via v(H)=v50*(H/50)^0.073096.
    """
    ws = ws50.loc[start:end]
    vH = ws["WS50M"].to_numpy(dtype=float) * (H / 50.0) ** 0.073096
    P = np.zeros_like(vH, dtype=float)
    # 2.5 <= v < 10
    m1 = (vH >= 2.5) & (vH < 10.0)
    P[m1] = 0.191619426414177 * (D ** 2) * (vH[m1] ** 3)
    # 10 <= v <= 24
    m2 = (vH >= 10.0) & (vH <= 24.0)
    P[m2] = 6.0
    return pd.Series(P, index=ws.index)


def minimal_turbines_for_day_window(P: pd.Series,
                                    demand_mw: float = 300.0,
                                    start_hour: int = 6,
                                    end_hour: int = 22) -> Tuple[Optional[int], Dict]:
    """Compute minimal integer N such that N*P_t >= demand for all t within [start_hour, end_hour].

    Returns (N_min or None if infeasible, info dict with diagnostics).
    """
    if P.empty:
        return None, {"reason": "empty power series in window"}

    hrs = P.index.hour
    mask = (hrs >= start_hour) & (hrs <= end_hour)
    P_day = P[mask]
    if P_day.empty:
        return None, {"reason": "no data within required hours"}

    # If any hour has zero power, infeasible under strict per-hour requirement
    zero_mask = (P_day <= 0.0)
    if zero_mask.any():
        return None, {
            "reason": "zero-power hours present in 06:00-22:00 window",
            "zero_hours": int(zero_mask.sum()),
            "first_zero": str(P_day.index[zero_mask][0])
        }

    # Required turbines per hour: ceil(demand / P_t)
    N_req = np.ceil(demand_mw / P_day.to_numpy())
    N_min = int(N_req.max())
    worst_idx = int(N_req.argmax())
    worst_time = P_day.index[worst_idx]
    worst_P = float(P_day.iloc[worst_idx])
    return N_min, {
        "worst_time": str(worst_time),
        "worst_P_per_turbine": worst_P,
    }


def minimal_turbines_for_reliability(P: pd.Series, reliability: float,
                                     demand_mw: float = 300.0,
                                     start_hour: int = 6,
                                     end_hour: int = 22) -> Tuple[Optional[int], Dict]:
    """Compute minimal N for which N*P_t >= demand holds for at least `reliability` fraction
    of hours within [start_hour, end_hour]. If the (1-reliability) quantile of P is 0, returns None.
    """
    hrs = P.index.hour
    mask = (hrs >= start_hour) & (hrs <= end_hour)
    P_day = P[mask]
    if P_day.empty:
        return None, {"reason": "no data within required hours"}
    q = np.quantile(P_day.to_numpy(), 1.0 - reliability)
    if q <= 0.0:
        return None, {"reason": f"quantile at {(1.0 - reliability):.1%} is 0 -> infeasible"}
    N = int(math.ceil(demand_mw / q))
    return N, {"quantile_power": q}


def farthest_point_subset(points: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Select k points spread out from a candidate set using farthest-point sampling.

    points: (N,2) array
    Returns indices of selected points (length k). If k >= N, returns all indices.
    """
    N = len(points)
    if k >= N:
        return np.arange(N)
    rng = np.random.default_rng(seed)
    # Start from a random point for better spread
    start = int(rng.integers(0, N))
    selected = [start]
    # Maintain min distance to selected set for each candidate
    d2min = np.full(N, np.inf, dtype=float)
    # Initialize distances to first point
    dx = points[:, 0] - points[start, 0]
    dy = points[:, 1] - points[start, 1]
    d2min = dx * dx + dy * dy
    d2min[start] = 0.0
    while len(selected) < k:
        # Choose the farthest remaining point
        idx = int(np.argmax(d2min))
        selected.append(idx)
        # Update min distances
        dx = points[:, 0] - points[idx, 0]
        dy = points[:, 1] - points[idx, 1]
        d2 = dx * dx + dy * dy
        d2min = np.minimum(d2min, d2)
        d2min[selected] = 0.0
    return np.array(selected, dtype=int)


def main():
    diameters = [80.0, 100.0, 120.0]
    H = 150.0
    forb = ForbiddenLines.create(L)

    # Data window
    start = '2024-08-01 00:00'
    end = '2025-08-01 00:00'

    # Load wind data
    ws_df = load_hourly_ws50(os.path.join('data', 'point.csv'))

    results = []
    for D in diameters:
        P = compute_power_series(ws_df, H, D, start, end)
        Nmin, info = minimal_turbines_for_day_window(P)
        results.append({
            'D': D,
            'Nmin': Nmin,
            'info': info,
            'P': P,
        })

    # Filter feasible
    feasible = [r for r in results if r['Nmin'] is not None]
    if not feasible:
        print("All diameters infeasible for strict 06:00–22:00 >=300MW over the selected year.")
        for r in results:
            print(f"D={int(r['D'])}m: infeasible -> {r['info']}")
        # Provide reliability-based alternatives
        reliabilities = [0.99, 0.95, 0.90]
        alt_rows = []
        for r in results:
            D = r['D']
            P = r['P']
            for rel in reliabilities:
                Nrel, info = minimal_turbines_for_reliability(P, rel)
                alt_rows.append({
                    'D': D,
                    'reliability': rel,
                    'N': Nrel,
                    'info': info,
                })
        # Show summary
        print("\nReliability-based minimal turbines (N) suggestions:")
        for row in alt_rows:
            D = int(row['D'])
            rel = row['reliability']
            if row['N'] is None:
                print(f"  D={D}m @ {int(rel*100)}%: infeasible -> {row['info']}")
            else:
                print(f"  D={D}m @ {int(rel*100)}%: N={row['N']} (quantile power={row['info']['quantile_power']:.3f} MW)")

        # Choose a demonstration layout for the best (smallest N) among feasible reliability cases
        feasible_alt = [row for row in alt_rows if row['N'] is not None]
        if feasible_alt:
            feasible_alt.sort(key=lambda x: (x['N'], -x['D'], -x['reliability']))
            pick = feasible_alt[0]
            D = pick['D']
            Nmin = pick['N']
            rel = pick['reliability']
            print(f"\nDemonstration layout: D={int(D)} m, N={Nmin}, reliability={int(rel*100)}%.")
            spacing = 5.0 * D
            X, Y, offset = best_hex_grid_in_allowed(L, spacing, forb, nx=14, ny=14)
            if X.size >= Nmin:
                centers_all = np.stack([X, Y], axis=1)
                sel_idx = farthest_point_subset(centers_all, Nmin, seed=11)
                centers = centers_all[sel_idx]
                figpath = os.path.join('figures', f'layout_q2_2_demo_D{int(D)}_{int(rel*100)}p.png')
                visualize_layout(D, centers, forb, figpath)
                print(f"Figure saved: {figpath}")
            else:
                print(f"Not enough allowed positions ({X.size}) to place {Nmin} turbines with spacing 5D.")
        return

    # Choose diameter with smallest Nmin; tie-breaker: larger D (fewer rows needed)
    feasible.sort(key=lambda r: (r['Nmin'], -r['D']))
    best = feasible[0]
    D = best['D']
    Nmin = best['Nmin']
    info = best['info']
    print(f"Chosen diameter D={int(D)} m with minimal turbines N*={Nmin}.")
    print(f"Worst hour per-turbine power={info['worst_P_per_turbine']:.3f} MW at {info['worst_time']}.")

    # Build candidate grid with spacing 5D
    spacing = 5.0 * D
    X, Y, offset = best_hex_grid_in_allowed(L, spacing, forb, nx=14, ny=14)
    if X.size < Nmin:
        print(f"Layout infeasible: only {X.size} positions available with spacing 5D, but need {Nmin} turbines.")
        return

    centers_all = np.stack([X, Y], axis=1)
    sel_idx = farthest_point_subset(centers_all, Nmin, seed=7)
    centers = centers_all[sel_idx]

    # Visualize
    figpath = os.path.join('figures', f'layout_q2_2_D{int(D)}.png')
    visualize_layout(D, centers, forb, figpath)

    print("Summary:")
    print(f"  D = {int(D)} m")
    print(f"  Minimal turbines N* = {Nmin}")
    print(f"  Grid spacing = {spacing:.1f} m (>=5D)")
    print(f"  Offset used = {offset}")
    print(f"  Figure saved: {figpath}")


if __name__ == "__main__":
    main()
