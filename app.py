"""Module 5: hoofdinterface van de Copernicus ERA5 Klimaatapp (Streamlit)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from components.charts import create_climate_fan_chart, create_monthly_quantile_chart
from utils.cds_client import get_cds_client
from utils.data_fetcher import fetch_era5_monthly_data
from utils.stats import calculate_monthly_climatology

# --------------------------------------------------------------------------
# Pagina-instellingen (moet het eerste Streamlit-commando zijn)
# --------------------------------------------------------------------------
st.set_page_config(page_title="Copernicus Klimaatapp", page_icon="🌍", layout="wide")

# --------------------------------------------------------------------------
# Configuratie: locaties en variabelen
# --------------------------------------------------------------------------
PRESET_LOCATIONS: dict[str, tuple[float, float]] = {
    "Amsterdam": (52.37, 4.90),
    "Madrid": (40.42, -3.70),
    "Nairobi": (-1.29, 36.82),
    "Tokyo": (35.68, 139.69),
    "Sydney": (-33.87, 151.21),
}

MIN_YEAR, MAX_YEAR = 1979, 2025  # laatste volledige kalenderjaar
DEFAULT_PERIOD = (1991, 2020)    # standaard WMO-klimaatnormaalperiode


@dataclass(frozen=True)
class VariableConfig:
    """Beschrijft hoe een variabele wordt opgehaald en omgerekend."""

    cds_variables: tuple[str, ...]            # wat we bij de CDS opvragen
    title: str                                # titel in grafieken
    unit: str                                 # eenheid ná omrekening
    transform: Callable[[xr.Dataset], xr.DataArray]  # omrekening naar 1 reeks
    note: str = ""                            # uitleg onder de grafiek


VARIABLES: dict[str, VariableConfig] = {
    "2m Temperatuur": VariableConfig(
        ("2m_temperature",), "Temperatuur", "°C",
        lambda ds: ds["t2m"],  # al in °C gezet door data_fetcher
    ),
    "Totale Neerslag": VariableConfig(
        ("total_precipitation",), "Neerslag", "mm/maand",
        # ERA5: gemiddelde dagsom in meter -> mm per maand
        lambda ds: ds["tp"] * 1000 * ds["time"].dt.days_in_month,
    ),
    "Windsnelheid (10 m)": VariableConfig(
        ("10m_u_component_of_wind", "10m_v_component_of_wind"),
        "Windsnelheid", "m/s",
        lambda ds: np.hypot(ds["u10"], ds["v10"]),
        note=(
            "Windsnelheid is berekend uit de maandgemiddelde u- en v-component. "
            "Dit is de snelheid van de gemiddelde windvector en ligt daardoor "
            "lager dan de werkelijke gemiddelde windsnelheid."
        ),
    ),
    "Zonnestraling": VariableConfig(
        ("surface_solar_radiation_downwards",), "Zonnestraling", "W/m²",
        # J/m² per dag -> gemiddeld vermogen in W/m²
        lambda ds: ds["ssrd"] / 86400,
    ),
    "Luchtdruk (zeeniveau)": VariableConfig(
        ("mean_sea_level_pressure",), "Luchtdruk", "hPa",
        lambda ds: ds["msl"] / 100,
    ),
}


# --------------------------------------------------------------------------
# Hulpfuncties
# --------------------------------------------------------------------------
def show_chart(fig) -> None:
    """Toon een Plotly-figuur op volle breedte (werkt met oude en nieuwe Streamlit)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:  # noqa: BLE001 - oudere Streamlit-versie
        st.plotly_chart(fig, use_container_width=True)


def build_sidebar() -> tuple[str, float, float, tuple[int, int], str, bool]:
    """Teken de sidebar en geef de invoer terug.

    Returns:
        (locatienaam, breedtegraad, lengtegraad, (startjaar, eindjaar),
        variabelelabel, knop_ingedrukt)
    """
    st.sidebar.header("⚙️ Instellingen")

    st.sidebar.subheader("📍 Locatie")
    mode = st.sidebar.radio(
        "Kies invoermethode", ["Preset Locatie", "Handmatige Coördinaten"],
        label_visibility="collapsed",
    )
    if mode == "Preset Locatie":
        name = st.sidebar.selectbox("Stad", list(PRESET_LOCATIONS))
        lat, lon = PRESET_LOCATIONS[name]
        st.sidebar.caption(f"Breedtegraad {lat:.2f}°, lengtegraad {lon:.2f}°")
    else:
        lat = st.sidebar.number_input(
            "Breedtegraad (°)", min_value=-90.0, max_value=90.0,
            value=52.37, step=0.01, format="%.2f",
        )
        lon = st.sidebar.number_input(
            "Lengtegraad (°)", min_value=-180.0, max_value=180.0,
            value=4.90, step=0.01, format="%.2f",
        )
        name = f"{lat:.2f}°, {lon:.2f}°"

    st.sidebar.subheader("📅 Tijdsperiode")
    period = st.sidebar.slider(
        "Jaren (van t/m)", MIN_YEAR, MAX_YEAR, DEFAULT_PERIOD
    )
    if period[1] - period[0] + 1 < 10:
        st.sidebar.warning(
            "Minder dan 10 jaar: percentielen zijn dan weinig betrouwbaar. "
            "Voor klimaatstatistiek worden meestal 30 jaar gebruikt."
        )

    st.sidebar.subheader("🌡️ Parameter")
    variable = st.sidebar.selectbox("Klimaatvariabele", list(VARIABLES))

    st.sidebar.divider()
    clicked = st.sidebar.button("Genereer Klimaatanalyse", type="primary")
    return name, float(lat), float(lon), (int(period[0]), int(period[1])), variable, clicked


def run_analysis(
    name: str, lat: float, lon: float, period: tuple[int, int], variable: str
) -> None:
    """Haal data op, bereken de klimatologie en bewaar het resultaat in session_state."""
    cfg = VARIABLES[variable]
    years = tuple(range(period[0], period[1] + 1))

    # Controleer eerst de credentials: toont zelf een duidelijke fout + stopt bij problemen.
    get_cds_client()

    try:
        with st.spinner(
            "Data ophalen uit Copernicus... Dit kan bij de eerste keer enkele minuten "
            "duren (de aanvraag staat in de wachtrij van de CDS)."
        ):
            ds = fetch_era5_monthly_data(lat, lon, years, cfg.cds_variables)
            series = cfg.transform(ds)
            series = series.rename("value")
            df = calculate_monthly_climatology(xr.Dataset({"value": series}), "value")
    except KeyError as exc:
        st.error(
            f"De verwachte variabele ontbreekt in de gedownloade data: {exc}. "
            "Probeer een andere parameter of probeer het later opnieuw."
        )
        return
    except ValueError as exc:
        st.error(f"Ongeldige invoer: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - alle API/netwerkfouten netjes tonen
        st.error(
            "Het ophalen van de data is mislukt.\n\n"
            f"**Technische melding:** `{exc}`\n\n"
            "**Mogelijke oplossingen:**\n"
            "- Controleer je `url` en `key` in de Streamlit *Secrets* (of `.streamlit/secrets.toml`).\n"
            "- Heb je de licentie van de ERA5-dataset geaccepteerd op de CDS-website?\n"
            "- Kies een kortere periode; de CDS kan bij drukte of grote aanvragen weigeren.\n"
            "- Probeer het over enkele minuten opnieuw."
        )
        st.session_state.pop("analysis", None)
        return

    if df[["mean", "p10", "p50", "p90"]].isna().all().all():
        st.warning("Er zijn geen bruikbare waarden gevonden voor deze selectie.")
        return

    st.session_state["analysis"] = {
        "name": name,
        "lat": lat,
        "lon": lon,
        "grid_lat": float(ds["latitude"]),
        "grid_lon": float(ds["longitude"]),
        "period": period,
        "variable": variable,
        "cfg": cfg,
        "df": df,
    }


def render_results(result: dict) -> None:
    """Toon metrics en de drie tabs op basis van een opgeslagen resultaat."""
    cfg: VariableConfig = result["cfg"]
    df: pd.DataFrame = result["df"]
    start, end = result["period"]

    # --- Status-block ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Locatie", result["name"])
    c2.metric("Gridpunt (ERA5)", f"{result['grid_lat']:.2f}°, {result['grid_lon']:.2f}°")
    c3.metric("Periode", f"{start} – {end}")
    c4.metric("Parameter", f"{cfg.title} ({cfg.unit})")
    st.caption(
        "ERA5 heeft een resolutie van ca. 0,25° (±28 km): het getoonde gridpunt is het "
        "dichtstbijzijnde punt bij je locatie."
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Maandelijkse Boxplots", "🌫️ Klimaat Bandgrafiek", "📋 Datatabel & Export"]
    )

    with tab1:
        show_chart(create_monthly_quantile_chart(df, cfg.title, cfg.unit))
        st.caption("Whisker = P10 tot P90, vierkant = mediaan, ruitje = gemiddelde.")
        if cfg.note:
            st.info(cfg.note)

    with tab2:
        show_chart(create_climate_fan_chart(df, cfg.title, cfg.unit))
        if cfg.note:
            st.info(cfg.note)

    with tab3:
        table = df.rename(
            columns={
                "month": "Maand nr",
                "month_name": "Maand",
                "mean": f"Gemiddelde ({cfg.unit})",
                "p50": f"P50 / Mediaan ({cfg.unit})",
                "p10": f"P10 ({cfg.unit})",
                "p90": f"P90 ({cfg.unit})",
            }
        ).round(2)
        st.dataframe(table, hide_index=True)

        fmt = st.radio(
            "CSV-formaat",
            ["Nederlands Excel (; en decimale komma)", "Internationaal (, en decimale punt)"],
            horizontal=True,
        )
        if fmt.startswith("Nederlands"):
            csv = table.to_csv(index=False, sep=";", decimal=",")
        else:
            csv = table.to_csv(index=False)
        st.download_button(
            "⬇️ Download als CSV",
            data=csv.encode("utf-8-sig"),
            file_name=f"klimaat_{cfg.title.lower()}_{start}-{end}.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Hoofdprogramma
# --------------------------------------------------------------------------
def main() -> None:
    """Bouw de pagina op."""
    st.title("🌍 Copernicus Klimaatapp")
    st.subheader("Maandelijkse klimaatstatistieken op basis van ERA5-reanalysedata")
    st.write(
        "Kies in de zijbalk een locatie, een periode en een klimaatvariabele. "
        "De app berekent per kalendermaand het gemiddelde en de percentielen "
        "P10, P50 (mediaan) en P90 over alle gekozen jaren."
    )

    name, lat, lon, period, variable, clicked = build_sidebar()

    if clicked:
        run_analysis(name, lat, lon, period, variable)

    if "analysis" in st.session_state:
        render_results(st.session_state["analysis"])
        st.caption(
            "Wijzig je invoer in de zijbalk en klik opnieuw op *Genereer Klimaatanalyse* "
            "om de resultaten te verversen."
        )
    else:
        st.info(
            "👈 Nog geen analyse gestart. Kies in de zijbalk een locatie "
            "(preset of eigen coördinaten), een periode en een parameter, en klik "
            "op **Genereer Klimaatanalyse**."
        )


main()
