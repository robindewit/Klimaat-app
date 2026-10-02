"""Module 3: klimatologische statistieken per kalendermaand."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

MONTH_NAMES: list[str] = [
    "Jan", "Feb", "Mrt", "Apr", "Mei", "Jun",
    "Jul", "Aug", "Sep", "Okt", "Nov", "Dec",
]

# Korte NetCDF-namen van ERA5 <-> vriendelijke/officiële CDS-namen.
_VARIABLE_ALIASES: dict[str, str] = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "total_precipitation": "tp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
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
        f"Variabele '{variable_name}' niet gevonden. "
        f"Beschikbaar in dataset: {list(ds.data_vars)}"
    )


def calculate_monthly_climatology(ds: xr.Dataset, variable_name: str) -> pd.DataFrame:
    """Bereken per kalendermaand gemiddelde, P50, P10 en P90 over alle jaren.

    Args:
        ds: Dataset met een ``time``-dimensie (maandelijkse data, één locatie).
        variable_name: Naam van de variabele, bv. ``"t2m"`` of ``"2m_temperature"``.

    Returns:
        DataFrame met kolommen
        ``['month', 'month_name', 'mean', 'p50', 'p10', 'p90']`` en 12 rijen.
        Maanden zonder data krijgen NaN.
    """
    var = _resolve_variable(ds, variable_name)
    da = ds[var].squeeze(drop=True)

    # Eventuele overgebleven ruimtelijke dimensies middelen tot één reeks.
    extra_dims = [d for d in da.dims if d != "time"]
    if extra_dims:
        da = da.mean(dim=extra_dims, skipna=True)

    grouped = da.groupby("time.month")

    # skipna=True: missende waarden worden genegeerd.
    mean = grouped.mean(dim="time", skipna=True)
    p50 = grouped.quantile(0.50, dim="time", skipna=True)
    p10 = grouped.quantile(0.10, dim="time", skipna=True)
    p90 = grouped.quantile(0.90, dim="time", skipna=True)

    # Zorg dat alle 12 maanden aanwezig zijn, ook als er geen data voor is.
    months = np.arange(1, 13)
    df = pd.DataFrame(
        {
            "month": months,
            "month_name": MONTH_NAMES,
            "mean": mean.reindex(month=months).values.astype(float),
            "p50": p50.reindex(month=months).values.astype(float),
            "p10": p10.reindex(month=months).values.astype(float),
            "p90": p90.reindex(month=months).values.astype(float),
        }
    )
    return df
