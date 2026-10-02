"""Module 2: ERA5 maandgemiddelden ophalen (alle parameters in één call), cachen en omrekenen."""
from __future__ import annotations

import os
import tempfile
import zipfile
from dataclasses import dataclass

import numpy as np
import streamlit as st
import xarray as xr

from utils.cds_client import get_cds_client

DATASET_NAME = "reanalysis-era5-single-levels-monthly-means"
PRODUCT_TYPE = "monthly_averaged_reanalysis"
BBOX_MARGIN = 0.25  # graden; ERA5-grid is 0.25°

# Alle MVP-parameters die in één CDS-call worden opgevraagd.
CDS_VARIABLES: list[str] = [
    "2m_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "mean_sea_level_pressure",
]


@dataclass(frozen=True)
class ParameterInfo:
    """Beschrijving van een weer te geven klimaatparameter (na omrekening)."""

    key: str         # variabelenaam in de Dataset
    tab_label: str   # tab-titel in de UI
    title: str       # titel in grafieken/tabellen/PDF
    unit: str        # eenheid na omrekening
    note: str = ""   # toelichting


PARAMETERS: tuple[ParameterInfo, ...] = (
    ParameterInfo("t2m", "Thermisch (°C)", "Temperatuur", "°C"),
    ParameterInfo(
        "tp", "Neerslag (mm)", "Neerslag", "mm/maand",
        "Neerslag is de totale hoeveelheid per kalendermaand (ERA5-dagsom in m, omgerekend naar mm en vermenigvuldigd met het aantal dagen).",
    ),
    ParameterInfo(
        "wind_speed", "Wind (m/s)", "Windsnelheid", "m/s",
        "Windsnelheid is berekend als sqrt(u² + v²) uit de maandgemiddelde windcomponenten. Dit is de snelheid van de gemiddelde windvector en ligt lager dan de werkelijke gemiddelde windsnelheid.",
    ),
    ParameterInfo(
        "ssrd", "Zonnestraling (J/m²)", "Zonnestraling", "J/m²/dag",
        "Zonnestraling is de gemiddelde dagsom aan inkomende kortgolvige straling aan het oppervlak (J/m² per dag).",
    ),
    ParameterInfo("msl", "Luchtdruk (hPa)", "Luchtdruk", "hPa"),
)


def _build_request(lat: float, lon: float, years: tuple[int, ...]) -> dict:
    """Stel het CDS-requestdictionary samen (alle variabelen tegelijk)."""
    return {
        "product_type": PRODUCT_TYPE,
        "variable": CDS_VARIABLES,
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
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _open_downloaded(path: str) -> xr.Dataset:
    """Open de download (NetCDF of zip met NetCDF's) en laad alles in het geheugen."""
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
        return ds.load()
    finally:
        ds.close()


def _select_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """Harmoniseer dimensies en kies het dichtstbijzijnde gridpunt."""
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    if "expver" in ds.dims:  # ERA5 (1) en ERA5T (5) combineren
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    ds = ds.drop_vars(["expver", "number"], errors="ignore")

    target_lon = lon % 360 if (float(ds["longitude"].max()) > 180 and lon < 0) else lon
    return ds.sel(latitude=lat, longitude=target_lon, method="nearest")


def _convert_units(ds: xr.Dataset) -> xr.Dataset:
    """Eenheidsconversies en afgeleide variabelen (windsnelheid)."""
    ds = ds.copy()

    if "t2m" in ds and ds["t2m"].attrs.get("units", "K") == "K":
        ds["t2m"] = ds["t2m"] - 273.15  # Kelvin -> °C
        ds["t2m"].attrs = {"units": "°C", "long_name": "2m temperatuur"}

    if "tp" in ds and ds["tp"].attrs.get("units", "m") == "m":
        # m per dag (maandgemiddelde) -> mm per maand
        ds["tp"] = ds["tp"] * 1000.0 * ds["time"].dt.days_in_month
        ds["tp"].attrs = {"units": "mm/maand", "long_name": "Totale neerslag"}

    if "msl" in ds and ds["msl"].attrs.get("units", "Pa") == "Pa":
        ds["msl"] = ds["msl"] / 100.0  # Pa -> hPa
        ds["msl"].attrs = {"units": "hPa", "long_name": "Luchtdruk op zeeniveau"}

    if "ssrd" in ds:
        ds["ssrd"].attrs = {"units": "J/m²/dag", "long_name": "Zonnestraling"}

    if "u10" in ds and "v10" in ds:
        ds["wind_speed"] = np.hypot(ds["u10"], ds["v10"])  # sqrt(u² + v²)
        ds["wind_speed"].attrs = {"units": "m/s", "long_name": "Windsnelheid (10 m)"}
    return ds


@st.cache_data(
    ttl=86400,
    show_spinner="Data ophalen uit Copernicus Climate Data Store...",
)
def fetch_era5_monthly_data(lat: float, lon: float, years: tuple[int, ...]) -> xr.Dataset:
    """Haal alle ERA5-parameters in één CDS-call op voor het dichtstbijzijnde gridpunt.

    Args:
        lat: Breedtegraad (-90..90).
        lon: Lengtegraad (-180..180).
        years: Jaren, bv. ``(1991, ..., 2020)`` (tuple i.v.m. cache-hashing).

    Returns:
        Dataset met dimensie ``time`` en de variabelen ``t2m`` (°C), ``tp``
        (mm/maand), ``u10``, ``v10``, ``wind_speed`` (m/s), ``ssrd`` (J/m²/dag)
        en ``msl`` (hPa).
    """
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Coördinaten buiten bereik (lat -90..90, lon -180..180).")
    if not years:
        raise ValueError("Geef minimaal één jaar op.")

    request = _build_request(lat, lon, years)
    client = get_cds_client()

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp.close()  # handle sluiten (nodig op Windows); CDS schrijft zelf het bestand
    try:
        client.retrieve(DATASET_NAME, request, tmp.name)
        ds = _open_downloaded(tmp.name)
        return _convert_units(_select_point(ds, lat, lon))
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
