"""Module 2: ERA5 maandgemiddelden ophalen, cachen en opschonen."""
from __future__ import annotations

import os
import tempfile
import zipfile

import streamlit as st
import xarray as xr

from utils.cds_client import get_cds_client

DATASET_NAME = "reanalysis-era5-single-levels-monthly-means"
PRODUCT_TYPE = "monthly_averaged_reanalysis"
BBOX_MARGIN = 0.25  # graden; ERA5-grid is 0.25° dus ~1 gridpunt rondom

# Vriendelijke namen / aliassen -> officiële CDS-parameternamen.
VARIABLE_MAP: dict[str, str] = {
    "2m_temperature": "2m_temperature",
    "2m_dewpoint_temperature": "2m_dewpoint_temperature",
    "total_precipitation": "total_precipitation",
    "10m_u_component_of_wind": "10m_u_component_of_wind",
    "10m_v_component_of_wind": "10m_v_component_of_wind",
    "surface_solar_radiation_downwards": "surface_solar_radiation_downwards",
    "mean_sea_level_pressure": "mean_sea_level_pressure",
    # Handige aliassen
    "temperature": "2m_temperature",
    "dewpoint": "2m_dewpoint_temperature",
    "precipitation": "total_precipitation",
    "wind_u": "10m_u_component_of_wind",
    "wind_v": "10m_v_component_of_wind",
    "solar_radiation": "surface_solar_radiation_downwards",
    "pressure": "mean_sea_level_pressure",
}

# Korte NetCDF-variabelenamen die van Kelvin naar Celsius moeten.
_KELVIN_SHORT_NAMES = ("t2m", "d2m")


def _map_variables(variables: tuple[str, ...]) -> list[str]:
    """Zet vriendelijke namen om naar CDS-parameternamen (ontdubbeld, op volgorde)."""
    if not variables:
        raise ValueError("Geef minimaal één variabele op.")
    mapped: list[str] = []
    for name in variables:
        key = name.strip().lower()
        if key not in VARIABLE_MAP:
            raise ValueError(
                f"Onbekende variabele '{name}'. Ondersteund: {sorted(VARIABLE_MAP)}"
            )
        cds_name = VARIABLE_MAP[key]
        if cds_name not in mapped:
            mapped.append(cds_name)
    return mapped


def _build_request(
    lat: float, lon: float, years: tuple[int, ...], variables: list[str]
) -> dict:
    """Stel het CDS-requestdictionary samen."""
    return {
        "product_type": PRODUCT_TYPE,
        "variable": variables,
        "year": [str(y) for y in sorted(set(years))],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        # [Noord, West, Zuid, Oost]
        "area": [
            round(lat + BBOX_MARGIN, 4),
            round(lon - BBOX_MARGIN, 4),
            round(lat - BBOX_MARGIN, 4),
            round(lon + BBOX_MARGIN, 4),
        ],
        # Nieuwe CDS-API (2024+) gebruikt 'data_format' i.p.v. 'format'.
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _open_downloaded(path: str) -> xr.Dataset:
    """Open het download (NetCDF of zip met NetCDF's), laad alles in het geheugen.

    De nieuwe CDS levert soms een zip met meerdere .nc-bestanden (bv. wanneer
    ERA5 en ERA5T gemengd worden). Die worden hier samengevoegd.
    """
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path) as zf:
                nc_files = [n for n in zf.namelist() if n.endswith(".nc")]
                zf.extractall(tmpdir, members=nc_files)
            parts: list[xr.Dataset] = []
            for n in nc_files:
                with xr.open_dataset(os.path.join(tmpdir, n), engine="netcdf4") as d:
                    parts.append(d.load())
        return xr.merge(parts, compat="override") if len(parts) > 1 else parts[0]

    ds = xr.open_dataset(path, engine="netcdf4")
    try:
        return ds.load()  # volledig in geheugen
    finally:
        ds.close()  # file handle vrijgeven


def _normalize(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """Harmoniseer dimensies, kies gridpunt en converteer eenheden."""
    # Nieuwe CDS gebruikt 'valid_time' i.p.v. 'time'.
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    # ERA5 (expver=1) en ERA5T (expver=5) combineren indien aanwezig.
    if "expver" in ds.dims:
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    ds = ds.drop_vars(["expver", "number"], errors="ignore")

    # Longitude-conventie (0..360 vs -180..180) afstemmen.
    target_lon = lon
    if float(ds["longitude"].max()) > 180 and lon < 0:
        target_lon = lon % 360

    ds = ds.sel(latitude=lat, longitude=target_lon, method="nearest")

    # Kelvin -> Celsius (alleen als de eenheid nog 'K' is, voorkomt dubbele conversie).
    for short in _KELVIN_SHORT_NAMES:
        if short in ds.data_vars and ds[short].attrs.get("units", "K") == "K":
            attrs = dict(ds[short].attrs)
            ds[short] = ds[short] - 273.15
            attrs["units"] = "°C"
            ds[short].attrs = attrs
    return ds


@st.cache_data(
    ttl=86400,
    show_spinner="Data ophalen uit Copernicus Climate Data Store...",
)
def fetch_era5_monthly_data(
    lat: float,
    lon: float,
    years: tuple[int, ...],
    variables: tuple[str, ...],
) -> xr.Dataset:
    """Haal ERA5-maandgemiddelden op voor het dichtstbijzijnde gridpunt.

    Args:
        lat: Breedtegraad in graden (-90..90).
        lon: Lengtegraad in graden (-180..180).
        years: Jaren, bv. ``(2020, 2021, 2022)``. Tuple i.v.m. cache-hashing.
        variables: Vriendelijke of officiële CDS-variabelenamen (tuple).

    Returns:
        ``xr.Dataset`` met dimensie ``time`` voor het gekozen gridpunt.
        Temperaturen (``t2m``, ``d2m``) staan in °C.
    """
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Coördinaten buiten bereik (lat -90..90, lon -180..180).")
    if not years:
        raise ValueError("Geef minimaal één jaar op.")

    request = _build_request(lat, lon, years, _map_variables(variables))
    client = get_cds_client()

    # Tijdelijk bestand; handle meteen sluiten (nodig op Windows) en door CDS laten vullen.
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp.close()
    try:
        client.retrieve(DATASET_NAME, request, tmp.name)
        ds = _open_downloaded(tmp.name)
        return _normalize(ds, lat, lon)
    finally:
        try:
            os.remove(tmp.name)  # cleanup, ook bij fouten
        except OSError:
            pass
