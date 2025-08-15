import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


L = 20000.0  # square size in meters (0..L for both x and y)


@dataclass
class ForbiddenLines:
    """Defines the two forbidden half-planes via line equations y = m x + b.

    - Forbidden 1: above the line through A=(0, L/2) and B=(2L/3, L).
    - Forbidden 2: below the line through C=(L/4, 0) and D=(L, L/4).

    Points exactly on the lines are ALLOWED (included), per user spec.
    """
    m1: float
    b1: float
    m2: float
    b2: float

    @staticmethod
    def create(L: float) -> "ForbiddenLines":
        # Line 1 through (0, L/2) and (2L/3, L)
        m1 = (L - (L / 2.0)) / (2.0 * L / 3.0 - 0.0)  # = 0.75
        b1 = L / 2.0
        # Line 2 through (L/4, 0) and (L, L/4)
        m2 = (L / 4.0 - 0.0) / (L - L / 4.0)  # = 1/3
        b2 = -m2 * (L / 4.0)  # = -L/12
        return ForbiddenLines(m1=m1, b1=b1, m2=m2, b2=b2)

    def y1(self, x: np.ndarray) -> np.ndarray:
        return self.m1 * x + self.b1

    def y2(self, x: np.ndarray) -> np.ndarray:
        return self.m2 * x + self.b2

    def is_allowed(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Allowed region = inside square and NOT in forbidden half-planes.

        Forbidden1: y > y1(x)  -> allowed is y <= y1(x)
        Forbidden2: y < y2(x)  -> allowed is y >= y2(x)
        Boundaries inclusive.
        """
        y1_val = self.y1(x)
        y2_val = self.y2(x)
        return (
            (x >= 0.0) & (x <= L) & (y >= 0.0) & (y <= L)
            & (y <= y1_val + 1e-9)
            & (y >= y2_val - 1e-9)
        )


def generate_hex_grid(L: float, spacing: float, offset: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a hexagonal lattice of points within [0,L]x[0,L].

    spacing: center-to-center spacing along x for even rows (s = 5D)
    row pitch: s_y = s * sqrt(3) / 2
    offset: (ox, oy) applied before tiling
    """
    s = spacing
    sy = s * math.sqrt(3.0) / 2.0
    ox, oy = offset

    # y rows
    ys = np.arange(oy, L + 1e-9, sy)
    xs_list = []
    ys_list = []
    for i, y in enumerate(ys):
        # staggered hex grid: odd rows shifted by s/2
        xstart = ox + (s / 2.0 if (i % 2 == 1) else 0.0)
        xs = np.arange(xstart, L + 1e-9, s)
        xs_list.append(xs)
        ys_list.append(np.full_like(xs, y))
    if not xs_list:
        return np.array([]), np.array([])
    X = np.concatenate(xs_list)
    Y = np.concatenate(ys_list)
    # keep within box strictly (inclusive boundaries)
    mask = (X >= 0.0) & (X <= L) & (Y >= 0.0) & (Y <= L)
    return X[mask], Y[mask]


def best_hex_grid_in_allowed(L: float, spacing: float, forb: ForbiddenLines,
                             nx: int = 12, ny: int = 12) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Search offsets within one hex cell to maximize number of allowed points.

    Returns: (X_allowed, Y_allowed, chosen_offset)
    """
    s = spacing
    sy = s * math.sqrt(3.0) / 2.0
    best_count = -1
    best_offset = (0.0, 0.0)
    best_points = (np.array([]), np.array([]))

    for ix in range(nx):
        ox = (ix / nx) * s
        for iy in range(ny):
            oy = (iy / ny) * sy
            X, Y = generate_hex_grid(L, s, (ox, oy))
            if X.size == 0:
                continue
            mask = forb.is_allowed(X, Y)
            cnt = int(np.count_nonzero(mask))
            if cnt > best_count:
                best_count = cnt
                best_offset = (ox, oy)
                best_points = (X[mask], Y[mask])

    return best_points[0], best_points[1], best_offset


def min_pairwise_distance(points: np.ndarray) -> float:
    """Compute minimum pairwise distance among points (Nx2 array). O(N^2) is OK for typical counts here."""
    if len(points) < 2:
        return float("inf")
    # Efficient vectorized distance computation by blocks if needed; for now simple O(N^2)
    min_d2 = float("inf")
    N = len(points)
    for i in range(N):
        dx = points[i, 0] - points[i + 1:, 0]
        dy = points[i, 1] - points[i + 1:, 1]
        d2 = np.min(dx * dx + dy * dy) if i + 1 < N else float("inf")
        if d2 < min_d2:
            min_d2 = d2
    return math.sqrt(min_d2) if np.isfinite(min_d2) else float("inf")


def load_hourly_ws50(filepath: str) -> pd.DataFrame:
    """Load data/point.csv and construct a datetime index from the first 4 columns.

    Expects a column named 'WS50M' for wind speed at 50m.
    """
    # Detect header line index where actual CSV header starts with 'YEAR,'
    header_idx = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('YEAR'):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("Could not find 'YEAR' header line in data/point.csv")

    df = pd.read_csv(filepath, skiprows=header_idx, header=0)
    # Try to standardize column names
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    # Assume first four are Year, Month, Day, Hour
    ycol, mcol, dcol, hcol = cols[0], cols[1], cols[2], cols[3]
    if "WS50M" not in df.columns:
        raise ValueError("WS50M column not found in data/point.csv after header parsing")

    dt = pd.to_datetime(
        dict(year=df[ycol], month=df[mcol], day=df[dcol], hour=df[hcol]),
        errors="coerce"
    )
    df["datetime"] = dt
    df = df.set_index("datetime").sort_index()
    return df[["WS50M"]]


def compute_per_turbine_energy_mwh(ws50: pd.Series, H: float, D: float,
                                   start: str, end: str, inclusive: str = "both") -> float:
    """Compute per-turbine energy (MWh) over the time window [start, end] with inclusive control.

    P(v) in MW:
      0,  v < 2.5
      0.191619426414177 * D^2 * v^3, 2.5 <= v < 10
      6,  10 <= v <= 24
      0,  v > 24

    Sum hourly MW over hours -> MWh.
    """
    # Slice time series inclusively (loc is inclusive on both ends for strings)
    ws = ws50.loc[start:end]
    vH = ws["WS50M"].to_numpy(dtype=float) * (H / 50.0) ** 0.073096

    P = np.zeros_like(vH, dtype=float)
    # piecewise
    mask_1 = (vH >= 2.5) & (vH < 10.0)
    mask_2 = (vH >= 10.0) & (vH <= 24.0)
    P[mask_1] = 0.191619426414177 * (D ** 2) * (vH[mask_1] ** 3)
    P[mask_2] = 6.0
    # vH < 2.5 or vH > 24 => 0 already

    energy_mwh = float(P.sum())  # MW * 1h sums to MWh
    return energy_mwh


def visualize_layout(D: float, centers: np.ndarray, forb: ForbiddenLines, outpath: str):
    s = 5.0 * D
    r = 2.5 * D  # safety radius for visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_title(f"Layout for D={int(D)} m (spacing=5D={int(s)} m)")

    # Shade forbidden regions within the square using fill_between
    xs = np.linspace(0, L, 400)
    y1 = forb.y1(xs)
    y2 = forb.y2(xs)
    # Clip to [0, L] for plotting aesthetics
    y1c = np.clip(y1, 0, L)
    y2c = np.clip(y2, 0, L)

    # Forbidden 1: y > y1(x) up to top
    ax.fill_between(xs, y1c, L, color='0.85', step=None, alpha=0.8, label='Forbidden 1')
    # Forbidden 2: y < y2(x) down to bottom
    ax.fill_between(xs, 0, y2c, color='0.85', step=None, alpha=0.8, label='Forbidden 2')

    # Square border
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color='black', linewidth=1)

    # Turbine centers and safety circles
    if len(centers) > 0:
        ax.scatter(centers[:, 0], centers[:, 1], s=10, c='tab:blue', label='Turbines', zorder=3)
        # Draw a subset of circles to avoid huge rendering cost if many points
        # max_circles = 500
        # step = max(1, len(centers) // max_circles)
        # for i in range(0, len(centers), step):
        #     circ = plt.Circle((centers[i, 0], centers[i, 1]), r, color='tab:blue', fill=False, alpha=0.3, linewidth=0.5)
        #     ax.add_patch(circ)

    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def run_for_diameters():
    diameters = [80.0, 100.0, 120.0]
    H = 150.0
    forb = ForbiddenLines.create(L)

    # Load wind data once
    ws_df = load_hourly_ws50(os.path.join('data', 'point.csv'))
    start = '2024-08-01 00:00'
    end = '2025-08-01 00:00'  # inclusive per user

    results: List[Dict] = []

    for D in diameters:
        spacing = 5.0 * D
        X, Y, offset = best_hex_grid_in_allowed(L, spacing, forb, nx=14, ny=14)
        centers = np.stack([X, Y], axis=1) if X.size > 0 else np.zeros((0, 2))

        # Safety check: min distance >= spacing - tiny eps
        mind = min_pairwise_distance(centers)
        if not (math.isfinite(mind) and mind + 1e-6 >= spacing or len(centers) <= 1):
            # Very unlikely, but we can perform a simple greedy pruning fallback
            order = np.argsort(centers[:, 0] + 1e-6 * centers[:, 1])
            kept = []
            for idx in order:
                p = centers[idx]
                if all(np.hypot(p[0] - centers[j][0], p[1] - centers[j][1]) >= spacing - 1e-9 for j in kept):
                    kept.append(idx)
            centers = centers[kept]
            mind = min_pairwise_distance(centers)

        per_turbine_mwh = compute_per_turbine_energy_mwh(ws_df, H, D, start, end)
        total_mwh = per_turbine_mwh * len(centers)

        # Visualization
        figpath = os.path.join('figures', f'layout_D{int(D)}.png')
        visualize_layout(D, centers, forb, figpath)

        results.append({
            'D': D,
            'count': len(centers),
            'per_turbine_MWh': per_turbine_mwh,
            'total_MWh': total_mwh,
            'min_distance_m': mind,
            'offset': offset,
            'figure': figpath,
        })

    # Print a compact summary
    print("Diameter(m)  Count  MinDist(m)  PerTurbine(MWh)  Total(MWh)  Figure")
    for r in results:
        print(f"{int(r['D']):>10}  {r['count']:>5}  {r['min_distance_m']:>10.1f}  "
              f"{r['per_turbine_MWh']:>15.1f}  {r['total_MWh']:>10.1f}  {r['figure']}")


if __name__ == "__main__":
    run_for_diameters()
