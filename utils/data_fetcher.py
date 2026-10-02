"""Module 2: ERA5-data ophalen (maandgemiddelden, drukvlakken, uurdata) met parallelle CDS-calls.

Bronnen
-------
1. ``reanalysis-era5-single-levels-monthly-means``   -> maandgemiddelden (alle oppervlakteparameters)
2. ``reanalysis-era5-pressure-levels-monthly-means``  -> verticale profielen (Expert)
3. ``reanalysis-era5-single-levels-timeseries``       -> uurdata voor afgeleide indicatoren
   (Tx/Tn, warmtedagen, WBGT, windroos, IDF). Deze bron bevat maar een beperkte set
   parameters en kan door ECMWF tijdelijk uitgeschakeld zijn.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cdsapi
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from utils.cds_client import get_cds_credentials

DATASET_MONTHLY = "reanalysis-era5-single-levels-monthly-means"
DATASET_PRESSURE = "reanalysis-era5-pressure-levels-monthly-means"
DATASET_HOURLY = "reanalysis-era5-single-levels-timeseries"
BBOX_MARGIN = 0.25
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200]

# Aanvragen worden in groepen gesplitst: faalt één groep, dan blijft de rest bruikbaar.
MONTHLY_GROUPS: dict[str, list[str]] = {
    "core": [
        "2m_temperature", "2m_dewpoint_temperature",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "total_precipitation", "total_evaporation",
        "surface_solar_radiation_downwards",
    ],
    "aviation": ["instantaneous_10m_wind_gust", "visibility", "cloud_base_height"],
    "surface_expert": [
        "large_scale_precipitation", "convective_precipitation",
        "surface_latent_heat_flux", "surface_sensible_heat_flux",
        "boundary_layer_height",
        "convective_available_potential_energy", "convective_inhibition",
    ],
    "soil": (
        [f"volumetric_soil_water_layer_{i}" for i in range(1, 5)]
        + [f"soil_temperature_level_{i}" for i in range(1, 5)]
    ),
}
USER_GROUPS = ("core", "aviation")
EXPERT_GROUPS = ("surface_expert", "soil")

PRESSURE_VARIABLES = [
    "temperature", "specific_humidity",
    "u_component_of_wind", "v_component_of_wind", "geopotential",
]

HOURLY_VARIABLES = [
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "10m_wind_gust_since_previous_post_processing",
    "total_precipitation", "surface_solar_radiation_downwards",
]
# korte NetCDF-namen (varianten) -> vaste kolomnaam
_HOURLY_SHORT = {
    "t2m": ("t2m", "2t"), "d2m": ("d2m", "2d"), "u10": ("u10", "10u"),
    "v10": ("v10", "10v"), "fg10": ("fg10", "10fg", "i10fg"),
    "tp": ("tp",), "ssrd": ("ssrd",),
}


@dataclass
class FetchBundle:
    """Resultaat van alle downloads; ``errors`` bevat mislukte onderdelen."""

    monthly: xr.Dataset | None = None
    pressure: xr.Dataset | None = None
    hourly: pd.DataFrame | None = None
    errors: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Download-hulpfuncties
# --------------------------------------------------------------------------
def _open_downloaded(path: str) -> xr.Dataset:
    """Open NetCDF (of zip met NetCDF's), laad alles in het geheugen."""
    if zipfile.is_zipfile(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if n.endswith(".nc")]
                zf.extractall(tmpdir, members=names)
            parts = []
            for n in names:
                with xr.open_dataset(os.path.join(tmpdir, n), engine="netcdf4") as d:
                    parts.append(d.load())
        return xr.merge(parts, compat="override") if len(parts) > 1 else parts[0]
    ds = xr.open_dataset(path, engine="netcdf4")
    try:
        return ds.load()
    finally:
        ds.close()


def _download(dataset: str, request: dict, url: str, key: str) -> xr.Dataset:
    """Download naar een tijdelijk bestand, laad in geheugen en ruim op (thread-safe)."""
    client = cdsapi.Client(url=url, key=key, quiet=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp.close()
    try:
        client.retrieve(dataset, request, tmp.name)
        return _open_downloaded(tmp.name)
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass


def _area(lat: float, lon: float) -> list[float]:
    return [round(lat + BBOX_MARGIN, 4), round(lon - BBOX_MARGIN, 4),
            round(lat - BBOX_MARGIN, 4), round(lon + BBOX_MARGIN, 4)]


def _years(years: tuple[int, ...]) -> list[str]:
    return [str(y) for y in sorted(set(years))]


def _rename_time(ds: xr.Dataset) -> xr.Dataset:
    return ds.rename({"valid_time": "time"}) if "valid_time" in ds.dims or "valid_time" in ds.coords else ds


def _select_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """Harmoniseer dimensies en kies het dichtstbijzijnde gridpunt."""
    ds = _rename_time(ds)
    if "expver" in ds.dims:
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))
    ds = ds.drop_vars(["expver", "number"], errors="ignore")
    target_lon = lon % 360 if (float(ds["longitude"].max()) > 180 and lon < 0) else lon
    return ds.sel(latitude=lat, longitude=target_lon, method="nearest")


# --------------------------------------------------------------------------
# Eenheidsconversies
# --------------------------------------------------------------------------
def convert_monthly(ds: xr.Dataset) -> xr.Dataset:
    """Basis-eenheden: °C, mm/maand, W/m² (flux omhoog > 0), km, m/s, J/m²/dag."""
    ds = ds.copy()
    for n in ("t2m", "d2m", "stl1", "stl2", "stl3", "stl4"):
        if n in ds and ds[n].attrs.get("units", "K") == "K":
            ds[n] = ds[n] - 273.15
            ds[n].attrs = {"units": "°C"}
    days = ds["time"].dt.days_in_month
    for n in ("tp", "lsp", "cp"):  # m/dag -> mm/maand
        if n in ds:
            ds[n] = ds[n] * 1000.0 * days
            ds[n].attrs = {"units": "mm/maand"}
    if "e" in ds:  # ECMWF: verdamping is negatief -> positief maken
        ds["e"] = -ds["e"] * 1000.0 * days
        ds["e"].attrs = {"units": "mm/maand"}
    for n in ("slhf", "sshf"):  # J/m²/dag, neerwaarts > 0 -> W/m², opwaarts > 0
        if n in ds:
            ds[n] = -ds[n] / 86400.0
            ds[n].attrs = {"units": "W/m²"}
    if "vis" in ds:
        ds["vis"] = ds["vis"] / 1000.0
        ds["vis"].attrs = {"units": "km"}
    if "u10" in ds and "v10" in ds:  # snelheid van de gemiddelde windvector
        ds["wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    return ds


def convert_pressure(ds: xr.Dataset) -> xr.Dataset:
    """Profielen: t in °C, q in g/kg, z in gpm, plus dauwpunt ``td`` (°C) uit q en p."""
    ds = _rename_time(ds).copy()
    if "level" in ds.dims:
        ds = ds.rename({"level": "pressure_level"})
    if "q" in ds:
        q, p = ds["q"], ds["pressure_level"].astype(float)
        e = q * p / (0.622 + 0.378 * q)  # dampspanning in hPa
        ln = np.log(e / 6.112)
        ds["td"] = 243.5 * ln / (17.67 - ln)
        ds["q"] = q * 1000.0
    if "t" in ds:
        ds["t"] = ds["t"] - 273.15
    if "z" in ds:
        ds["z"] = ds["z"] / 9.80665
    return ds


def _hourly_frame(ds: xr.Dataset) -> pd.DataFrame:
    """Uurdata -> DataFrame met t2m/d2m (°C), u10/v10/fg10 (m/s), tp (mm/uur), ssrd (W/m²)."""
    ds = _rename_time(ds).squeeze(drop=True)
    cols: dict[str, pd.Series] = {}
    for target, variants in _HOURLY_SHORT.items():
        name = next((v for v in variants if v in ds.data_vars), None)
        if name:
            cols[target] = ds[name].to_series()
    if not cols:
        raise KeyError(f"Geen bekende variabelen in uurdata: {list(ds.data_vars)}")
    df = pd.DataFrame(cols).astype("float32")
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()].sort_index()
    for c in ("t2m", "d2m"):
        if c in df:
            df[c] = df[c] - 273.15
    if "tp" in df:
        df["tp"] = df["tp"] * 1000.0      # m per uur -> mm per uur
    if "ssrd" in df:
        df["ssrd"] = df["ssrd"] / 3600.0  # J/m² per uur -> W/m²
    return df


# --------------------------------------------------------------------------
# Jobs (draaien in threads; geen Streamlit-aanroepen!)
# --------------------------------------------------------------------------
def _job_monthly(group: str, lat: float, lon: float, years: tuple[int, ...], url: str, key: str) -> xr.Dataset:
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": MONTHLY_GROUPS[group],
        "year": _years(years),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": _area(lat, lon),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    return _select_point(_download(DATASET_MONTHLY, request, url, key), lat, lon)


def _job_pressure(lat: float, lon: float, years: tuple[int, ...], url: str, key: str) -> xr.Dataset:
    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": PRESSURE_VARIABLES,
        "pressure_level": [str(p) for p in PRESSURE_LEVELS],
        "year": _years(years),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": "00:00",
        "area": _area(lat, lon),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    ds = _select_point(_download(DATASET_PRESSURE, request, url, key), lat, lon)
    return convert_pressure(ds)


def _job_hourly(lat: float, lon: float, years: tuple[int, ...], url: str, key: str) -> pd.DataFrame:
    ys = sorted(set(years))
    request = {
        "variable": HOURLY_VARIABLES,
        # de dienst rondt af op het 0,25°-grid; zelf afronden voorkomt afwijzing
        "location": {"longitude": round(lon * 4) / 4, "latitude": round(lat * 4) / 4},
        "date": [f"{ys[0]}-01-01/{ys[-1]}-12-31"],
        "data_format": "netcdf",
    }
    return _hourly_frame(_download(DATASET_HOURLY, request, url, key))


# --------------------------------------------------------------------------
@st.cache_data(ttl=86400, show_spinner="Data ophalen uit Copernicus (parallelle aanvragen)...")
def fetch_bundle(lat: float, lon: float, years: tuple[int, ...], expert: bool) -> FetchBundle:
    """Haal alle benodigde ERA5-data parallel op.

    Args:
        lat, lon: Coördinaten.
        years: Jaren (tuple i.v.m. cache-hashing).
        expert: ``True`` voor de Expert Studie (extra parameters + drukvlakken).

    Returns:
        ``FetchBundle``. Mislukte onderdelen staan in ``errors`` (sleutels
        ``monthly:<groep>``, ``pressure``, ``hourly``); de rest blijft bruikbaar.

    Raises:
        ValueError / RuntimeError bij ongeldige invoer of als alles mislukt.
    """
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Coördinaten buiten bereik.")
    if not years:
        raise ValueError("Geef minimaal één jaar op.")

    url, key = get_cds_credentials()  # in de hoofdthread
    groups = USER_GROUPS + (EXPERT_GROUPS if expert else ())

    jobs: dict[str, tuple] = {
        f"monthly:{g}": (_job_monthly, (g, lat, lon, years, url, key)) for g in groups
    }
    if expert:
        jobs["pressure"] = (_job_pressure, (lat, lon, years, url, key))
    jobs["hourly"] = (_job_hourly, (lat, lon, years, url, key))

    bundle = FetchBundle()
    monthly_parts: list[xr.Dataset] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(fn, *args) for name, (fn, args) in jobs.items()}
        for name, fut in futures.items():
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                bundle.errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            if name.startswith("monthly:"):
                monthly_parts.append(result)
            elif name == "pressure":
                bundle.pressure = result
            else:
                bundle.hourly = result

    if monthly_parts:
        merged = xr.merge(monthly_parts, compat="override")
        bundle.monthly = convert_monthly(merged)

    if bundle.monthly is None and bundle.hourly is None:
        raise RuntimeError("Geen data kunnen ophalen: " + "; ".join(bundle.errors.values()))
    return bundle
