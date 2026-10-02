"""Module 3: klimatologische statistieken per kalendermaand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

MONTH_NAMES: list[str] = [
    "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun",
    "Jul", "Aug", "Sep", "Okt", "Nov", "Dec",
]

# Kolomnamen (één plek, zodat charts/PDF/app consistent blijven)
COL_MONTH_NR = "Maandnr"
COL_MONTH = "Maand"
COL_P10 = "P10"
COL_P50 = "P50 (Percentiel)"
COL_P90 = "P90"
COL_MEAN = "Gemiddelde"
VALUE_COLUMNS: list[str] = [COL_P10, COL_P50, COL_P90, COL_MEAN]

_VARIABLE_ALIASES: dict[str, str] = {
    "2m_temperature": "t2m",
    "total_precipitation": "tp",
    "surface_solar_radiation_downwards": "ssrd",
    "mean_sea_level_pressure": "msl",
}


def _resolve_variable(ds: xr.Dataset, variable_name: str) -> str:
    """Zoek de variabelenaam in de dataset (accepteert ook CDS-namen)."""
    if variable_name in ds.data_vars:
        return variable_name
    alias = _VARIABLE_ALIASES.get(variable_name)
    if alias and alias in ds.data_vars:
        return alias
    raise KeyError(
        f"Variabele '{variable_name}' niet gevonden. Beschikbaar: {list(ds.data_vars)}"
    )


def calculate_monthly_climatology(ds: xr.Dataset, variable_name: str) -> pd.DataFrame:
    """Bereken per kalendermaand P10, P50 (50% Percentiel), P90 en gemiddelde.

    Returns:
        DataFrame met kolommen
        ``['Maandnr', 'Maand', 'P10', 'P50 (Percentiel)', 'P90', 'Gemiddelde']``
        en 12 rijen. Maanden zonder data geven NaN.
    """
    da = ds[_resolve_variable(ds, variable_name)].squeeze(drop=True)
    extra_dims = [d for d in da.dims if d != "time"]
    if extra_dims:
        da = da.mean(dim=extra_dims, skipna=True)

    grouped = da.groupby("time.month")
    months = np.arange(1, 13)

    def col(result: xr.DataArray) -> np.ndarray:
        return result.reindex(month=months).values.astype(float)

    return pd.DataFrame(
        {
            COL_MONTH_NR: months,
            COL_MONTH: MONTH_NAMES,
            COL_P10: col(grouped.quantile(0.10, dim="time", skipna=True)),
            COL_P50: col(grouped.quantile(0.50, dim="time", skipna=True)),
            COL_P90: col(grouped.quantile(0.90, dim="time", skipna=True)),
            COL_MEAN: col(grouped.mean(dim="time", skipna=True)),
        }
    )


def number_decimals(df: pd.DataFrame) -> int:
    """Geschikt aantal decimalen op basis van de orde van grootte van de waarden."""
    values = df[VALUE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return 2
    peak = float(np.nanmax(np.abs(values)))
    if peak >= 1e5:
        return 0
    if peak >= 100:
        return 1
    return 2
