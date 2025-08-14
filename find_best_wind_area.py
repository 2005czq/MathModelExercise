#!/usr/bin/env python3
"""
Find the 50km x 50km region with the maximum 10-year average WS50M.

Assumptions and notes:
- Input CSV: data/windspeed.csv with columns LAT,LON,YEAR,MO,DY,WS50M
- We average WS50M over time for each (LAT,LON) point first, then compute the
  spatial mean within candidate 50km x 50km rectangles aligned with latitude/longitude axes.
- Degrees-to-km: ~111.32 km per degree latitude; longitude degrees scale with cos(latitude).
- The grid is assumed to be rectilinear in degrees (common for reanalysis data).
- No external dependencies required.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "windspeed.csv")

KM_PER_DEG_LAT = 111.32


@dataclass
class WindowResult:
    avg_ws: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    count_points: int
    center_lat: float
    center_lon: float


def read_location_means(csv_path: str) -> Tuple[Dict[Tuple[float, float], float], List[float], List[float]]:
    """Read CSV and return per-location mean WS50M plus sorted unique LAT/LON lists."""
    sums: Dict[Tuple[float, float], float] = defaultdict(float)
    counts: Dict[Tuple[float, float], int] = defaultdict(int)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Expect headers: LAT,LON,YEAR,MO,DY,WS50M
        for row in reader:
            try:
                lat = float(row["LAT"])  # type: ignore[index]
                lon = float(row["LON"])  # type: ignore[index]
                ws = float(row["WS50M"])  # type: ignore[index]
            except (ValueError, KeyError):
                # Skip malformed rows
                continue
            key = (lat, lon)
            sums[key] += ws
            counts[key] += 1

    means: Dict[Tuple[float, float], float] = {}
    for key, s in sums.items():
        c = counts[key]
        if c > 0:
            means[key] = s / c

    lats = sorted({lat for lat, _ in means.keys()})
    lons = sorted({lon for _, lon in means.keys()})
    return means, lats, lons


def infer_grid_step(values: List[float]) -> float:
    """Infer the minimal positive step between sorted coordinate values."""
    steps = []
    for a, b in zip(values, values[1:]):
        d = round(b - a, 10)
        if d > 0:
            steps.append(d)
    if not steps:
        return float("nan")
    # Use the minimum positive step to be conservative
    return min(steps)


def build_grid(
    means: Dict[Tuple[float, float], float], lats: List[float], lons: List[float]
) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Build 2D arrays for per-point mean and presence mask aligned to sorted lats/lons.
    Returns (grid_means, grid_mask) where grid_mask is 1 for present, 0 otherwise.
    """
    nlat, nlon = len(lats), len(lons)
    grid = [[0.0 for _ in range(nlon)] for _ in range(nlat)]
    mask = [[0 for _ in range(nlon)] for _ in range(nlat)]
    lon_index = {lon: j for j, lon in enumerate(lons)}
    lat_index = {lat: i for i, lat in enumerate(lats)}
    for (lat, lon), val in means.items():
        i = lat_index[lat]
        j = lon_index[lon]
        grid[i][j] = val
        mask[i][j] = 1
    return grid, mask


def prefix_sum_2d(a: List[List[float]]) -> List[List[float]]:
    """Compute 2D inclusive prefix sums."""
    n = len(a)
    m = len(a[0]) if n else 0
    ps = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row_sum = 0.0
        for j in range(1, m + 1):
            row_sum += a[i - 1][j - 1]
            ps[i][j] = ps[i - 1][j] + row_sum
    return ps


def rect_sum(ps: List[List[float]], i0: int, j0: int, i1: int, j1: int) -> float:
    """
    Sum over rectangle [i0,i1] x [j0,j1], inclusive, using 2D prefix sums.
    Assumes ps has shape (n+1, m+1).
    """
    return ps[i1 + 1][j1 + 1] - ps[i0][j1 + 1] - ps[i1 + 1][j0] + ps[i0][j0]


def best_window(
    lats: List[float],
    lons: List[float],
    grid_means: List[List[float]],
    grid_mask: List[List[int]],
    window_km: float = 50.0,
) -> WindowResult:
    nlat, nlon = len(lats), len(lons)
    if nlat == 0 or nlon == 0:
        raise RuntimeError("Empty grid")

    # Infer grid steps
    dlat = infer_grid_step(lats)
    dlon = infer_grid_step(lons)
    if not (dlat > 0 and dlon > 0):
        raise RuntimeError("Could not infer grid step from coordinates")

    # Precompute prefix sums for means and mask
    ps_mean = prefix_sum_2d(grid_means)
    ps_mask = prefix_sum_2d([[float(v) for v in row] for row in grid_mask])

    lat_width_deg = window_km / KM_PER_DEG_LAT

    best: Optional[WindowResult] = None

    for i0 in range(nlat):
        # Determine how many latitude indices fit into 50km
        span_lat = max(1, int(math.floor(lat_width_deg / dlat)) + 1)
        i1 = min(nlat - 1, i0 + span_lat - 1)

        # Use the center latitude of this candidate window to set lon span
        lat_center = lats[(i0 + i1) // 2]
        cosphi = math.cos(math.radians(lat_center))
        # Guard against poles (not relevant here but be safe)
        if cosphi < 1e-6:
            cosphi = 1e-6
        lon_width_deg = window_km / (KM_PER_DEG_LAT * cosphi)
        span_lon = max(1, int(math.floor(lon_width_deg / dlon)) + 1)

        for j0 in range(nlon):
            j1 = min(nlon - 1, j0 + span_lon - 1)
            cnt = rect_sum(ps_mask, i0, j0, i1, j1)
            if cnt <= 0:
                continue
            s = rect_sum(ps_mean, i0, j0, i1, j1)
            avg = s / cnt
            lat_min, lat_max = lats[i0], lats[i1]
            lon_min, lon_max = lons[j0], lons[j1]
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            res = WindowResult(
                avg_ws=avg,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                count_points=int(cnt),
                center_lat=center_lat,
                center_lon=center_lon,
            )
            if best is None or res.avg_ws > best.avg_ws:
                best = res

    if best is None:
        raise RuntimeError("No valid window found")
    return best


def main() -> None:
    print("Loading data...", flush=True)
    means, lats, lons = read_location_means(CSV_PATH)
    print(f"Unique points: {len(means)} | lats: {len(lats)} | lons: {len(lons)}")
    grid_means, grid_mask = build_grid(means, lats, lons)
    print("Scanning windows (50km x 50km)...", flush=True)
    best = best_window(lats, lons, grid_means, grid_mask, window_km=50.0)
    print("\nBest 50km x 50km region (axis-aligned):")
    print(f"  Center:     ({best.center_lat:.5f} N, {best.center_lon:.5f} E)")
    print(f"  Latitude:   [{best.lat_min:.5f}, {best.lat_max:.5f}] deg")
    print(f"  Longitude:  [{best.lon_min:.5f}, {best.lon_max:.5f}] deg")
    print(f"  Grid pts:   {best.count_points}")
    print(f"  Mean WS50M: {best.avg_ws:.3f} m/s")


if __name__ == "__main__":
    main()
