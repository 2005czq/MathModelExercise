import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "point.csv"
FIG_PATH = Path(__file__).parent / "figures" / "annual_energy_3d.png"


def load_ws50m(filepath: Path) -> pd.DataFrame:
    """Load NASA/POWER hourly CSV with preamble, returning DataFrame with datetime index and WS50M.

    Detects the header line (starting with 'YEAR,') and skips preamble robustly.
    """
    # Find the header line number dynamically
    header_line_idx = None
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip().startswith('YEAR,'):
                header_line_idx = i
                break
    if header_line_idx is None:
        raise ValueError("Could not locate header line starting with 'YEAR,' in the file.")

    df = pd.read_csv(
        filepath,
        skiprows=header_line_idx,  # skip everything before the header
        header=0,                  # the next line is the header
        na_values=[-999, "-999"],
    )
    # Ensure expected columns exist
    required = {"YEAR", "MO", "DY", "HR", "WS50M"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {filepath.name}: {missing}")

    # Construct datetime index from local time columns
    dt = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"]))
    out = pd.DataFrame({"datetime": dt, "WS50M": pd.to_numeric(df["WS50M"], errors="coerce")})
    out = out.dropna(subset=["WS50M"]).set_index("datetime").sort_index()
    return out


def wind_at_height(v50: pd.Series, H: float) -> pd.Series:
    """Scale 50 m wind speed to hub height H using power law exponent 0.073096.

    v(H) = v(50m) * (H/50)^0.073096
    """
    factor = (H / 50.0) ** 0.073096
    return v50 * factor


def power_curve(v: pd.Series, D: float) -> pd.Series:
    """Piecewise power curve as specified (units as given by coefficients).

    P(v) =
      0,                      v < 2.5
      0.191619426414177*D^2*v^3,  2.5 <= v < 10
      6,                      10 <= v <= 24
      0,                      v > 24
    """
    v = v.astype(float)
    c = 0.191619426414177
    p = np.zeros_like(v, dtype=float)
    mask_var = (v >= 2.5) & (v < 10)
    mask_flat = (v >= 10) & (v <= 24)
    p[mask_var] = c * (D ** 2) * (v[mask_var] ** 3)
    p[mask_flat] = 6.0
    return pd.Series(p, index=v.index)


def integrate_energy(power_series: pd.Series) -> float:
    """Integrate hourly power over time via summation.

    Assumes hourly time step; returns sum(P) for all hours.
    If the input has gaps or duplicates, it will still sum values as-is.
    """
    return float(power_series.sum())


def compute_annual_energy(ws50: pd.DataFrame, start: str, end: str,
                           D_values=(80, 100, 120), H_values=(90, 120, 150)) -> pd.DataFrame:
    """Compute annual energy for each combination of rotor diameter D and hub height H.

    Returns a DataFrame indexed by D with columns H containing energy values in MWh.
    """
    # Filter to requested window: [start, end)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    ws = ws50.loc[(ws50.index >= s) & (ws50.index < e), "WS50M"].copy()

    if ws.empty:
        raise ValueError("No WS50M data in the requested time range.")

    energies = pd.DataFrame(index=D_values, columns=H_values, dtype=float)
    for D in D_values:
        for H in H_values:
            vH = wind_at_height(ws, H)
            P = power_curve(vH, D)
            E = integrate_energy(P)  # hourly sum of power in kW
            E_MWh = E / 1000000.0     # convert kW to MW, then sum over hours to get MWh
            energies.loc[D, H] = E_MWh
    return energies


def plot_energy_3d(energies: pd.DataFrame, save_path: Path | None = None) -> None:
    """Plot a 3D bar chart for energy with axes (D index, H columns).

    The z-axis represents Annual Energy in MWh.
    """
    Ds = energies.index.to_list()
    Hs = energies.columns.to_list()

    # Map categorical positions to indices for clean 3D bars
    x_pos = np.arange(len(Ds))
    y_pos = np.arange(len(Hs))
    xx, yy = np.meshgrid(x_pos, y_pos, indexing='ij')
    x = xx.ravel()
    y = yy.ravel()
    z_bottom = np.zeros_like(x, dtype=float)
    dx = dy = 0.6  # bar width/depth

    # Heights (dz) from energies matrix in matching order
    # Divide by 1 million to convert Wh to MWh for plotting
    dz = np.array([energies.loc[D, H] for D in Ds for H in Hs], dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, z_bottom, dx, dy, dz, shade=True, color='#1f77b4', alpha=0.9)

    # Ticks and labels
    ax.set_xticks(x_pos + dx / 2)
    ax.set_xticklabels([str(d) for d in Ds])
    ax.set_xlabel('Rotor Diameter D (m)')

    ax.set_yticks(y_pos + dy / 2)
    ax.set_yticklabels([str(h) for h in Hs])
    ax.set_ylabel('Hub Height H (m)')

    ax.set_zlabel('Annual Energy (MWh)')
    ax.set_title('Annual Energy (MWh) vs. Rotor Diameter (m) and Hub Height (m)')

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)


def main():
    ws50 = load_ws50m(DATA_PATH)
    energies = compute_annual_energy(ws50, start="2024-08-01 00:00", end="2025-08-01 00:00")
    plot_energy_3d(energies, FIG_PATH)
    # Also print the table for quick inspection
    with pd.option_context('display.float_format', '{:,.3f}'.format):
        print("Annual energy (MWh):")
        print(energies)


if __name__ == "__main__":
    main()