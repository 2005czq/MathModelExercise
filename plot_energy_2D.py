import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "point.csv"
# Updated figure path for the new 2D plot
FIG_PATH = Path(__file__).parent / "figures" / "annual_energy_2d_grouped.png"


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
            E_MWh = E / 1000.0       # convert kWh to MWh
            energies.loc[D, H] = E_MWh
    return energies


def plot_energy_2d_grouped(energies: pd.DataFrame, save_path: Path | None = None) -> None:
    """Plot a 2D grouped bar chart for energy.

    Groups are by Rotor Diameter (D, from index).
    Bars within groups are by Hub Height (H, from columns).
    The y-axis represents Annual Energy in MWh.
    """
    D_values = energies.index
    H_values = energies.columns
    n_H = len(H_values)

    x = np.arange(len(D_values))  # the label locations for D groups
    width = 0.25  # the width of an individual bar

    fig, ax = plt.subplots(figsize=(12, 8), layout='constrained')

    # Calculate the offset for each bar within a group to center them
    for i, H in enumerate(H_values):
        offset = width * (i - (n_H - 1) / 2)
        energy_values = energies[H]
        rects = ax.bar(x + offset, energy_values, width, label=f'H = {H} m')
        ax.bar_label(rects, padding=3, fmt='{:,.0f}')

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel('Annual Energy (MWh)')
    ax.set_xlabel('Rotor Diameter D (m)')
    ax.set_title('Annual Energy (MWh) vs. Rotor Diameter and Hub Height')
    ax.set_xticks(x, D_values)
    ax.legend(title='Hub Height')

    # Adjust y-axis limit to make space for the labels on top of the bars
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    """Main function to run the analysis and generate the plot."""
    # Assuming the data file 'point.csv' exists in a 'data' subdirectory
    # For demonstration, let's create a dummy file if it doesn't exist.
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        print(f"'{DATA_PATH}' not found. Creating a dummy file for demonstration.")
        dummy_data = "HEADER LINE 1\nHEADER LINE 2\nYEAR,MO,DY,HR,WS50M\n"
        dummy_df = pd.DataFrame({
            'YEAR': 2024,
            'MO': np.repeat(np.arange(1, 13), 30*24)[:8760],
            'DY': np.tile(np.arange(1, 31), 12*24)[:8760],
            'HR': np.tile(np.arange(24), 365),
            'WS50M': np.random.uniform(2, 15, 8760)
        })
        dummy_data += dummy_df.to_csv(index=False)
        with open(DATA_PATH, 'w') as f:
            f.write(dummy_data)

    ws50 = load_ws50m(DATA_PATH)
    energies = compute_annual_energy(ws50, start="2024-01-01 00:00", end="2025-01-01 00:00")
    
    # Call the new 2D plotting function
    plot_energy_2d_grouped(energies, FIG_PATH)
    
    # Also print the table for quick inspection
    with pd.option_context('display.float_format', '{:,.1f}'.format):
        print("Annual energy (MWh):")
        print(energies)


if __name__ == "__main__":
    main()